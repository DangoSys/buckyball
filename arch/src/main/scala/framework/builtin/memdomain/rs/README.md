# 保留站模块 (Reservation Station)

## 概述

保留站模块实现了内存域的指令调度和乱序执行管理，位于 `framework/builtin/memdomain/rs` 路径下。该模块包含保留站 (Reservation Station) 和重排序缓冲区 (ROB) 的实现，支持内存指令的发射、执行和完成管理。

## 文件结构

```
rs/
├── reservationStation.scala  - 内存保留站实现
├── rob.scala                 - 重排序缓冲区实现
└── ringFifo.scala           - 环形 FIFO 实现 (未使用)
```

## 核心组件

### MemReservationStation - 内存保留站

内存保留站负责管理内存指令的调度和执行：

```scala
class MemReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val mem_decode_cmd_i = Flipped(new DecoupledIO(new MemDecodeCmd))
    val rs_rocc_o = new Bundle {
      val resp  = new DecoupledIO(new RoCCResponseBB()(p))
      val busy  = Output(Bool())
    }
    val issue_o     = new MemIssueInterface
    val commit_i    = new MemCommitInterface
  })
}
```

#### 接口定义

**输入接口**：
- `mem_decode_cmd_i`: 来自内存域解码器的指令
- `commit_i`: 来自内存加载器/存储器的完成信号

**输出接口**：
- `issue_o`: 向内存加载器/存储器发射指令
- `rs_rocc_o`: 向 RoCC 接口返回响应

#### 发射接口

```scala
class MemIssueInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ld = Decoupled(new MemRsIssue)    // 加载指令发射
  val st = Decoupled(new MemRsIssue)    // 存储指令发射
}
```

#### 完成接口

```scala
class MemCommitInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ld = Flipped(Decoupled(new MemRsComplete))    // 加载指令完成
  val st = Flipped(Decoupled(new MemRsComplete))    // 存储指令完成
}
```

### ROB - 重排序缓冲区

ROB 管理指令的顺序执行和乱序完成：

```scala
class ROB (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val alloc = Flipped(new DecoupledIO(new MemDecodeCmd))
    val issue = new DecoupledIO(new RobEntry)
    val complete = Flipped(new DecoupledIO(UInt(log2Up(b.rob_entries).W)))
    val empty = Output(Bool())
    val full  = Output(Bool())
  })
}
```

#### ROB 条目

```scala
class RobEntry(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd    = new MemDecodeCmd                    // 内存指令
  val rob_id = UInt(log2Up(b.rob_entries).W)      // ROB 标识符
}
```

#### 核心数据结构

```scala
val robFifo = Module(new Queue(new RobEntry, b.rob_entries))
val robIdCounter = RegInit(0.U(log2Up(b.rob_entries).W))
val robTable = Reg(Vec(b.rob_entries, Bool()))
```

- `robFifo`: FIFO 队列，维护指令顺序
- `robIdCounter`: ROB ID 计数器
- `robTable`: 完成状态表，跟踪指令完成状态

## 工作流程

### 指令分配

1. **接收指令**：从内存域解码器接收 `MemDecodeCmd`
2. **分配 ROB ID**：为指令分配唯一的 ROB ID
3. **入队操作**：将指令和 ROB ID 存入 ROB FIFO
4. **状态初始化**：在完成状态表中标记为未完成

```scala
robFifo.io.enq.valid       := io.alloc.valid
robFifo.io.enq.bits.cmd    := io.alloc.bits
robFifo.io.enq.bits.rob_id := robIdCounter

when(io.alloc.fire) {
  robIdCounter := robIdCounter + 1.U
  robTable(robIdCounter) := false.B
}
```

### 指令发射

1. **检查头部指令**：检查 ROB 头部的未完成指令
2. **类型分离**：根据指令类型分离加载和存储操作
3. **发射控制**：只有在对应执行单元就绪时才发射

```scala
io.issue_o.ld.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_load
io.issue_o.st.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_store

rob.io.issue.ready  := (rob.io.issue.bits.cmd.is_load && io.issue_o.ld.ready) ||
                       (rob.io.issue.bits.cmd.is_store && io.issue_o.st.ready)
```

### 指令完成

1. **完成仲裁**：使用仲裁器处理多个完成信号
2. **状态更新**：在完成状态表中标记指令为已完成
3. **乱序支持**：支持指令乱序完成

```scala
val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))
completeArb.io.in(0).valid  := io.commit_i.ld.valid
completeArb.io.in(1).valid  := io.commit_i.st.valid

when(io.complete.fire) {
  robTable(io.complete.bits) := true.B
}
```

### 头部指令管理

ROB 只发射头部的未完成指令：

```scala
val headEntry     = robFifo.io.deq.bits
val headCompleted = robTable(headEntry.rob_id)
io.issue.valid   := robFifo.io.deq.valid && !headCompleted
robFifo.io.deq.ready := io.issue.ready && !headCompleted
```

## 配置参数

### ROB 配置

通过 `CustomBuckyBallConfig` 配置 ROB 参数：

```scala
class CustomBuckyBallConfig extends Config((site, here, up) => {
  case "rob_entries" => 16    // ROB 条目数量
})
```

### 使用示例

```scala
// 创建内存保留站
val memRS = Module(new MemReservationStation())

// 连接指令输入
memRS.io.mem_decode_cmd_i <> memDecoder.io.cmd_out

// 连接发射接口
memLoader.io.req <> memRS.io.issue_o.ld
memStorer.io.req <> memRS.io.issue_o.st

// 连接完成接口
memRS.io.commit_i.ld <> memLoader.io.resp
memRS.io.commit_i.st <> memStorer.io.resp

// 连接 RoCC 响应
rocc.io.resp <> memRS.io.rs_rocc_o.resp
```

## 执行模型

### 顺序发射，乱序完成

- **顺序发射**：指令按程序顺序从 ROB 头部发射
- **乱序完成**：指令可以乱序完成，通过 ROB ID 跟踪
- **顺序提交**：指令按程序顺序提交 (当前实现中简化)

### 内存一致性

- **加载存储分离**：加载和存储指令分别处理
- **依赖检查**：通过 ROB 维护内存访问顺序
- **异常处理**：支持内存访问异常的处理

## 状态监控

### ROB 状态

```scala
val isEmpty = robTable.reduce(_ && _)    // 所有指令都已完成
val isFull = !robFifo.io.enq.ready      // ROB 已满

io.empty := isEmpty
io.full  := isFull
```

### 忙碌信号

```scala
io.rs_rocc_o.busy := !rob.io.empty      // ROB 非空时保留站忙碌
```

## 性能考虑

### 吞吐量优化

- 支持加载和存储指令并行发射
- 使用仲裁器处理多个完成信号
- 最小化 ROB 查找延迟

### 资源利用

- ROB 大小可配置，平衡性能和面积
- 完成状态表使用位向量，节省存储
- FIFO 队列提供高效的顺序管理

## 相关模块

- [内存域概览](../README.md) - 上层内存管理
- [内存控制器](../mem/README.md) - 内存加载器和存储器
- [DMA 引擎](../dma/README.md) - DMA 数据传输
