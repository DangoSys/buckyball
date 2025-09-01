# Memory Bank Implementation Module

## 一、Overview

该目录包含了 BuckyBall 架构中内存域(MemDomain)的核心存储单元实现，主要负责提供高性能的片上存储器组件。该模块位于 `arch/src/main/scala/framework/builtin/memdomain/mem` 路径下，在整个系统架构中扮演底层硬件存储抽象的角色。

该目录实现了三种主要的存储器组件：
- **SramBank**: 基础SRAM存储Bank，提供同步读写功能
- **AccBank**: 累加器存储Bank，支持读-修改-写操作
- **Scratchpad**: 暂存器模块，管理多个存储Bank并提供仲裁机制

这些模块在系统中处于 `framework.builtin.memdomain` 的底层，为上层的内存控制器(MemController)、内存加载器(MemLoader)和内存存储器(MemStorer)提供基础的存储服务。

## 二、代码结构

```
mem/
├── SramBank.scala     - 基础SRAMBank实现
├── AccBank.scala      - 累加器Bank实现
└── Scratchpad.scala   - 暂存器管理模块
```

### 文件依赖关系

**SramBank.scala** (基础层)
- 定义了基础的SRAM接口(`SramReadIO`, `SramWriteIO`)
- 提供同步读写存储器的底层实现
- 被AccBank和Scratchpad模块所使用

**AccBank.scala** (扩展层)
- 依赖SramBank作为底层存储
- 实现累加器流水线(`AccPipe`)和读请求路由器(`AccReadRouter`)
- 扩展了`SramWriteIO`为`AccWriteIO`以支持累加操作

**Scratchpad.scala** (管理层)
- 聚合多个SramBank和AccBank实例
- 实现DMA和执行单元之间的请求仲裁
- 依赖配置文件`BaseConfig`进行参数化

### 数据流向

```
执行单元/DMA → Scratchpad → AccBank/SramBank → 物理存储器
                    ↓
                仲裁机制处理多路访问请求
```

## 三、模块详细说明

### SramBank.scala

**主要功能**: 提供基础的同步读写SRAM存储器实现

**关键组件**:

1. **接口定义**:
```scala
class SramReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class SramWriteReq(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val mask = Vec(mask_len, Bool())
  val data = UInt(w.W)
}
```

2. **核心逻辑**:
```scala
val mem = SyncReadMem(n, Vec(mask_len, mask_elem))

// 读写冲突仲裁
assert(!(io.read.req.valid && io.write.req.valid), 
       "SramBank: Read and write requests is not allowed at the same time")

io.read.req.ready := !io.write.req.valid
io.write.req.ready := !io.read.req.valid
```

**输入输出**:
- 输入: 读/写请求，包含地址、数据、掩码信息
- 输出: 读响应数据，带有延迟的有效信号
- 边缘情况: 不允许同周期读写同一Bank

**依赖项**: Chisel3 SyncReadMem，framework.builtin.util.Util

### AccBank.scala

**主要功能**: 实现支持累加操作的存储Bank

**关键组件**:

1. **累加流水线(AccPipe)**:
```scala
when (io.write_in.is_acc || RegNext(io.write_in.is_acc)) {
  // Stage 1: 读请求
  io.read.req.valid := io.write_in.req.valid
  
  // Stage 2: 累加运算
  val acc_data = data_reg + io.read.resp.bits.data
  
  // Stage 3: 写回
  io.write_out.req.bits.data := acc_data
}
```

2. **读请求路由器(AccReadRouter)**:
```scala
val req_arbiter = Module(new Arbiter(new SramReadReq(n), 2))
req_arbiter.io.in(0) <> io.read_in2.req  // 高优先级
req_arbiter.io.in(1) <> io.read_in1.req  // 低优先级

// 响应分发
val resp_to_in1 = RegNext(req_arbiter.io.chosen === 1.U && req_arbiter.io.out.fire)
```

**输入输出**:
- 输入: 带累加标志的写请求，读请求
- 输出: 累加结果写入底层SRAM
- 边缘情况: 流水线背压处理，读写请求冲突仲裁

**依赖项**: SramBank模块，Chisel3 Arbiter

### Scratchpad.scala

**主要功能**: 管理多个存储Bank，提供统一的暂存器接口

**关键组件**:

1. **Bank实例化**:
```scala
val spad_mems = Seq.fill(sp_banks) { Module(new SramBank(
  spad_bank_entries, spad_w, aligned_to, sp_singleported
)) }

val acc_mems = Seq.fill(acc_banks) { Module(new AccBank(
  acc_bank_entries, acc_w, aligned_to, sp_singleported
)) }
```

2. **请求仲裁机制**:
```scala
// 读请求仲裁：优先级 exec > dma
val exec_read_sel = exec_read_req.valid
val main_read_sel = main_read_req.valid && !exec_read_sel

// 响应分发
val resp_to_main = RegNext(main_read_sel && bank.io.read.req.fire)
val resp_to_exec = RegNext(exec_read_sel && bank.io.read.req.fire)
```

**输入输出**:
- 输入: DMA和执行单元的读写请求
- 输出: 仲裁后的存储器访问
- 边缘情况: 确保OpA和OpB不同时访问同一Bank

**依赖项**: BaseConfig配置，SramBank和AccBank模块，RocketChip tile参数

## 四、附加信息

### 注意事项

1. **单端口限制**: 配置中强制使用单端口SRAM(`sp_singleported = true`)，不允许同周期读写操作

2. **仲裁优先级**: 在所有模块中，执行单元(exec)的请求优先级高于DMA请求

3. **流水线设计**: AccBank采用三级流水线设计(读-累加-写)，需要考虑数据依赖和背压处理

4. **配置参数化**: 所有模块都支持通过BaseConfig进行参数化配置，包括Bank数量、容量、数据位宽等

5. **断言检查**: 代码中包含多个运行时断言，用于检测非法的并发访问和配置错误

6. **掩码支持**: 支持按字节粒度的写掩码操作，掩码长度由数据位宽和对齐要求计算得出