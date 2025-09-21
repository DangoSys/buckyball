# 向量处理单元 (Vector Processing Unit)

## 概述

向量处理单元是 BuckyBall 框架中的专用计算加速器，位于 `prototype/vector` 路径下。该模块实现了完整的向量处理流水线，包括控制单元、加载单元、执行单元和存储单元，支持向量数据的并行处理。

## 文件结构

```
vector/
├── VecUnit.scala         - 向量处理单元顶层模块
├── VecCtrlUnit.scala     - 向量控制单元
├── VecLoadUnit.scala     - 向量加载单元
├── VecEXUnit.scala       - 向量执行单元
├── VecStoreUnit.scala    - 向量存储单元
├── bond/                 - 绑定和同步机制
├── op/                   - 向量操作实现
├── thread/               - 线程管理
└── warp/                 - 线程束管理
```

## 核心组件

### VecUnit - 向量处理单元顶层

VecUnit 是向量处理器的顶层模块，集成了所有子单元：

```scala
class VecUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // 连接到Scratchpad的SRAM读写接口
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
    // 连接到Accumulator的读写接口
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })
}
```

#### 接口说明

**命令接口**：
- `cmdReq`: 来自保留站的向量指令请求
- `cmdResp`: 向保留站返回的完成响应

**存储接口**：
- `sramRead/sramWrite`: 连接到 Scratchpad 的读写接口
- `accRead/accWrite`: 连接到 Accumulator 的读写接口

### VecCtrlUnit - 向量控制单元

向量控制单元负责指令解码和流水线控制：

```scala
class VecCtrlUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle{
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp_o = Decoupled(new BallRsComplete)

    val ctrl_ld_o = Decoupled(new ctrl_ld_req)
    val ctrl_st_o = Decoupled(new ctrl_st_req)
    val ctrl_ex_o = Decoupled(new ctrl_ex_req)

    val cmdResp_i = Flipped(Valid(new Bundle {val commit = Bool()}))
  })
}
```

#### 控制状态

```scala
val rob_id_reg    = RegInit(0.U(log2Up(b.rob_entries).W))
val iter          = RegInit(0.U(10.W))
val op1_bank      = RegInit(0.U(2.W))
val op1_bank_addr = RegInit(0.U(12.W))
val op2_bank_addr = RegInit(0.U(12.W))
val op2_bank      = RegInit(0.U(2.W))
val wr_bank       = RegInit(0.U(2.W))
val wr_bank_addr  = RegInit(0.U(12.W))
val is_acc        = RegInit(false.B)
```

### 数据流架构

向量处理单元采用流水线架构，数据流如下：

```
指令输入 → VecCtrlUnit → 控制信号分发
                ↓
        VecLoadUnit (加载数据)
                ↓
        VecEXUnit (执行计算)
                ↓
        VecStoreUnit (存储结果)
                ↓
            完成响应
```

#### 模块连接

```scala
// 控制单元
val VecCtrlUnit = Module(new VecCtrlUnit)
VecCtrlUnit.io.cmdReq <> io.cmdReq
io.cmdResp <> VecCtrlUnit.io.cmdResp_o

// 加载单元
val VecLoadUnit = Module(new VecLoadUnit)
VecLoadUnit.io.ctrl_ld_i <> VecCtrlUnit.io.ctrl_ld_o

// 执行单元
val VecEX = Module(new VecEXUnit)
VecEX.io.ctrl_ex_i <> VecCtrlUnit.io.ctrl_ex_o
VecEX.io.ld_ex_i <> VecLoadUnit.io.ld_ex_o

// 存储单元
val VecStoreUnit = Module(new VecStoreUnit)
VecStoreUnit.io.ctrl_st_i <> VecCtrlUnit.io.ctrl_st_o
VecStoreUnit.io.ex_st_i <> VecEX.io.ex_st_o
```

## 存储系统集成

### Scratchpad 连接

向量处理单元通过多个 Bank 连接到 Scratchpad：

```scala
for (i <- 0 until b.sp_banks) {
  io.sramRead(i).req <> VecLoadUnit.io.sramReadReq(i)
  VecLoadUnit.io.sramReadResp(i) <> io.sramRead(i).resp
}
```

### Accumulator 连接

执行结果通过存储单元写入 Accumulator：

```scala
for (i <- 0 until b.acc_banks) {
  io.accWrite(i) <> VecStoreUnit.io.accWrite(i)
}
```

## 配置参数

### 向量配置

通过 `CustomBuckyBallConfig` 配置向量处理器参数：

```scala
class CustomBuckyBallConfig extends Config((site, here, up) => {
  case "veclane" => 16              // 向量通道数
  case "sp_banks" => 4              // Scratchpad Bank 数
  case "acc_banks" => 2             // Accumulator Bank 数
  case "spad_bank_entries" => 1024  // 每个 Bank 的条目数
  case "acc_bank_entries" => 512    // Accumulator 条目数
})
```

### 数据位宽

```scala
val spad_w = b.veclane * b.inputType.getWidth  // Scratchpad 位宽
val acc_w = b.outputType.getWidth              // Accumulator 位宽
```

## 使用方法

### 创建向量处理单元

```scala
val vecUnit = Module(new VecUnit())

// 连接命令接口
vecUnit.io.cmdReq <> reservationStation.io.issue
reservationStation.io.complete <> vecUnit.io.cmdResp

// 连接存储系统
for (i <- 0 until sp_banks) {
  scratchpad.io.read(i) <> vecUnit.io.sramRead(i)
  scratchpad.io.write(i) <> vecUnit.io.sramWrite(i)
}

for (i <- 0 until acc_banks) {
  accumulator.io.read(i) <> vecUnit.io.accRead(i)
  accumulator.io.write(i) <> vecUnit.io.accWrite(i)
}
```

### 向量指令格式

向量指令通过 `BallRsIssue` 接口传递：

```scala
class BallRsIssue extends Bundle {
  val cmd = new Bundle {
    val iter = UInt(10.W)           // 迭代次数
    val op1_bank = UInt(2.W)        // 操作数1的Bank
    val op1_bank_addr = UInt(12.W)  // 操作数1的地址
    val op2_bank = UInt(2.W)        // 操作数2的Bank
    val op2_bank_addr = UInt(12.W)  // 操作数2的地址
    val wr_bank = UInt(2.W)         // 写入Bank
    val wr_bank_addr = UInt(12.W)   // 写入地址
  }
  val rob_id = UInt(log2Up(rob_entries).W)
}
```

## 执行模型

### 流水线执行

1. **指令解码**：VecCtrlUnit 解码向量指令
2. **数据加载**：VecLoadUnit 从 Scratchpad 加载操作数
3. **向量计算**：VecEXUnit 执行向量运算
4. **结果存储**：VecStoreUnit 将结果写入 Accumulator
5. **完成响应**：向保留站返回完成信号

### 并行处理

- **多通道并行**：支持多个向量通道并行计算
- **Bank 级并行**：多个存储 Bank 支持并行访问
- **流水线重叠**：不同阶段可以重叠执行

## 子模块说明

### 绑定机制 (Bond)
提供线程间的同步和数据绑定功能，支持生产者-消费者模式的数据传递。

### 向量操作 (Op)
实现具体的向量计算操作，包括算术运算、逻辑运算和特殊函数。

### 线程管理 (Thread)
提供线程抽象和管理功能，支持不同类型的向量线程。

### 线程束管理 (Warp)
实现线程束的组织和调度，支持大规模并行计算。

## 性能特性

- **高并行度**：支持多通道向量并行处理
- **流水线化**：多级流水线提高吞吐量
- **存储优化**：多 Bank 存储系统减少访问冲突
- **灵活配置**：支持不同的向量长度和数据类型

## 相关模块

- [绑定机制](bond/README.md) - 线程同步和数据绑定
- [向量操作](op/README.md) - 具体的计算操作实现
- [线程管理](thread/README.md) - 线程抽象和管理
- [线程束管理](warp/README.md) - 线程束组织和调度
- [原型加速器概览](../README.md) - 上层加速器框架
