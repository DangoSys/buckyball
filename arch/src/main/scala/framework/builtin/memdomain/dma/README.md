# DMA Engine Implementation Module

## 一、Overview

该目录包含了 BuckyBall 架构中内存域(MemDomain)的DMA引擎实现，主要负责提供高性能的内存数据传输服务。该模块位于 `arch/src/main/scala/framework/builtin/memdomain/dma` 路径下，在整个系统架构中扮演内存域与外部存储器之间的数据传输桥梁角色。

该目录实现了两种主要的DMA组件：
- **BBStreamReader**: 流式数据读取器，支持批量从外部内存读取数据到片上存储
- **BBStreamWriter**: 流式数据写入器，支持批量从片上存储写入数据到外部内存
- **LocalAddr**: 本地地址管理工具，用于处理暂存器(Scratchpad)和累加器(Accumulator)的地址映射

这些模块在系统中处于 `framework.builtin.memdomain` 的数据传输层，为上层的内存加载器(MemLoader)和内存存储器(MemStorer)提供底层的DMA传输服务，并通过TileLink协议与外部内存系统进行通信。

## 二、代码结构

```
dma/
├── DMA.scala         - 流式DMA读写器实现
└── LocalAddr.scala   - 本地地址管理工具
```

### 文件依赖关系

**DMA.scala** (核心传输层)
- 定义了`BBReadRequest`、`BBReadResponse`、`BBWriteRequest`、`BBWriteResponse`等DMA传输接口
- 实现了`BBStreamReader`和`BBStreamWriter`两个核心DMA引擎
- 依赖TileLink协议进行外部内存访问
- 集成TLB虚拟地址转换功能

**LocalAddr.scala** (地址管理层)
- 定义了`LocalAddr` Bundle，用于统一管理本地存储器地址
- 支持暂存器(SP)和累加器(ACC)两种地址类型的区分和转换
- 提供地址运算、比较和越界检查功能
- 被上层MemLoader、MemStorer等模块广泛使用

### 数据流向

```
外部内存 ← TileLink ← BBStreamWriter ← MemStorer ← MemDomain ← 执行单元
外部内存 → TileLink → BBStreamReader → MemLoader → MemDomain → 执行单元
                            ↓
                    LocalAddr地址管理
                            ↓
                    暂存器/累加器寻址
```

## 三、模块详细说明

### DMA.scala

**主要功能**: 提供基于TileLink协议的流式DMA数据传输实现

**关键组件**:

#### 1. DMA请求响应接口定义

```scala
class BBReadRequest()(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)  // 虚拟地址
  val len = UInt(16.W)                 // 读取长度（字节）
  val status = new MStatus             // 处理器状态
}

class BBWriteRequest(dataWidth: Int)(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)      // 虚拟地址
  val data = UInt(dataWidth.W)             // 写入数据
  val len = UInt(16.W)                     // 写入长度（字节）
  val mask = UInt((dataWidth / 8).W)       // 字节掩码
  val status = new MStatus                 // 处理器状态
}
```

#### 2. BBStreamReader核心逻辑

```scala
// 状态机定义
val s_idle :: s_req_new_block :: Nil = Enum(2)
val state = RegInit(s_idle)

// 字节计数管理
val bytesRequested = Reg(UInt(16.W))  // 已发出请求的字节数
val bytesReceived = Reg(UInt(16.W))   // 已接收响应的字节数
val bytesLeft = req.len - bytesRequested

// TileLink请求构造
val get = edge.Get(
  fromSource = xactId,
  toAddress = 0.U,
  lgSize = log2Ceil(beatBytes).U
)._2
```

#### 3. TLB地址转换管道

```scala
class TLBundleAWithInfo extends Bundle {
  val tl_a = tl.a.bits.cloneType
  val vaddr = Output(UInt(vaddrBits.W))
  val status = Output(new MStatus)
}

// TLB请求处理
io.tlb.req.bits.tlb_req.vaddr := tlb_q.io.deq.bits.vaddr
io.tlb.req.bits.tlb_req.cmd := M_XRD  // 读操作
io.tlb.req.bits.status := tlb_q.io.deq.bits.status
```

#### 4. BBStreamWriter写入控制

```scala
// 选择合适的Put操作类型
val use_put_full = req.mask === ~0.U(beatBytes.W)
val selected_put = Mux(use_put_full, putFull, putPartial)

// 响应处理
io.resp.valid := tl.d.valid && edge.last(tl.d)
io.resp.bits.done := true.B
```

**输入输出**:
- 输入: 虚拟地址读写请求，包含地址、长度、数据、掩码
- 输出: 数据响应流，带有last标志和地址计数器
- 边缘情况: TLB缺失处理，事务ID管理，流水线背压

**依赖项**: TileLink协议，RocketChip TLB，Chisel3队列模块

### LocalAddr.scala

**主要功能**: 统一管理暂存器和累加器的本地地址映射

**关键组件**:

#### 1. 地址结构定义

```scala
class LocalAddr(sp_banks: Int, sp_bank_entries: Int, acc_banks: Int, acc_bank_entries: Int) extends Bundle {
  val is_acc_addr = Bool()         // 是否为累加器地址
  val accumulate = Bool()          // 是否执行累加操作
  val read_full_acc_row = Bool()   // 是否读取完整累加器行
  val data = UInt(memAddrBits.W)   // 实际地址数据
}
```

#### 2. 地址分解函数

```scala
// 暂存器地址分解
def sp_bank(dummy: Int = 0) = if (spAddrBits == spBankRowBits) 0.U 
                             else data(spAddrBits - 1, spBankRowBits)
def sp_row(dummy: Int = 0) = data(spBankRowBits - 1, 0)

// 累加器地址分解
def acc_bank(dummy: Int = 0) = if (accAddrBits == accBankRowBits) 0.U 
                              else data(accAddrBits - 1, accBankRowBits)
def acc_row(dummy: Int = 0) = data(accBankRowBits - 1, 0)
```

#### 3. 地址运算操作

```scala
// 地址加法运算
def +(other: UInt) = {
  val result = WireInit(this)
  result.data := data + other
  result
}

// 带溢出检查的加法
def add_with_overflow(other: UInt): Tuple2[LocalAddr, Bool] = {
  val sum = data +& other
  val overflow = Mux(is_acc_addr, sum(accAddrBits), sum(spAddrBits))
  (result, overflow)
}
```

#### 4. 特殊地址处理

```scala
// 垃圾地址检查
def is_garbage(dummy: Int = 0) = is_acc_addr && accumulate && read_full_acc_row && 
                                data.andR && garbage_bit.asBool

// 生成垃圾地址
def make_this_garbage(dummy: Int = 0): Unit = {
  is_acc_addr := true.B
  accumulate := true.B
  read_full_acc_row := true.B
  data := ~(0.U(maxAddrBits.W))
}
```

**输入输出**:
- 输入: Bank配置参数，地址数据
- 输出: Bank索引、行索引、地址比较结果
- 边缘情况: 地址溢出、下溢检查，垃圾地址处理

**依赖项**: Chisel3基础库，要求Bank entries为2的幂次

## 四、附加信息

### 注意事项

1. **地址对齐要求**: DMA操作需要考虑TileLink协议的对齐要求，BBStreamReader使用固定的beatBytes大小进行传输

2. **事务ID管理**: 两个DMA引擎都实现了事务ID的分配和回收机制，支持多个并发内存访问请求

3. **TLB集成**: DMA引擎完全集成了虚拟地址转换功能，支持用户态和内核态的内存访问

4. **流水线设计**: 读写器都采用流水线设计，包含地址转换、TileLink请求、响应处理等多个阶段

5. **错误处理**: 实现了TLB缺失处理，但未包含重试机制，依赖上层软件处理访问失败

6. **性能优化**: BBStreamWriter支持完整写和部分写两种模式，根据掩码自动选择最优的TileLink操作类型

7. **地址约束**: LocalAddr要求Bank entries必须为2的幂次，这简化了地址计算但限制了配置灵活性

8. **配置参数化**: DMA引擎支持通过构造参数配置并发事务数、数据位宽、最大传输字节数等关键参数