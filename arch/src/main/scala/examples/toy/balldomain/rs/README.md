# 保留站和重排序缓冲区 (Reservation Station & ROB)

## 概述

该模块实现了 BuckyBall 系统中的保留站（Reservation Station）和重排序缓冲区（ROB），用于支持乱序执行和指令调度。保留站负责管理指令的发射和完成，而 ROB 确保指令按程序顺序提交，维护处理器的精确异常语义。

## 二、文件结构

```
rs/
├── reservationStation.scala  - 保留站实现
└── rob.scala                - 重排序缓冲区实现
```

## 三、核心组件

### BallReservationStation - Ball域保留站

保留站是连接指令解码器和执行单元的关键组件，负责：

**主要功能**：
- 接收来自 Ball 域解码器的指令
- 根据指令类型分发到不同的执行单元
- 管理指令的发射和完成状态
- 生成 RoCC 响应

**支持的执行单元**：
- **ball1**: VecUnit（向量处理单元）
- **ball2**: BBFP（浮点处理单元）
- **ball3**: im2col（图像处理加速器）
- **ball4**: transpose（矩阵转置加速器）

**接口设计**：
```scala
class BallReservationStation extends Module {
  val io = IO(new Bundle {
    // 指令输入
    val ball_decode_cmd_i = Flipped(DecoupledIO(new BallDecodeCmd))

    // RoCC 响应输出
    val rs_rocc_o = new Bundle {
      val resp = DecoupledIO(new RoCCResponseBB)
      val busy = Output(Bool())
    }

    // 执行单元接口
    val issue_o = new BallIssueInterface    // 发射接口
    val commit_i = new BallCommitInterface  // 完成接口
  })
}
```

**指令分发逻辑**：
```scala
// 根据 bid (Ball ID) 分发指令
io.issue_o.ball1.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 1.U  // VecUnit
io.issue_o.ball2.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 2.U  // BBFP
io.issue_o.ball3.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 3.U  // im2col
io.issue_o.ball4.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 4.U  // transpose
```

### ROB - 重排序缓冲区

ROB 实现了指令的顺序管理和乱序完成支持：

**设计特点**：
- 使用 FIFO 队列维护指令顺序
- 使用完成状态表跟踪指令执行状态
- 支持乱序完成但顺序发射
- 提供 ROB ID 用于指令标识

**核心数据结构**：
```scala
class RobEntry extends Bundle {
  val cmd = new BallDecodeCmd           // 指令内容
  val rob_id = UInt(log2Up(rob_entries).W)  // ROB 标识符
}
```

**状态管理**：
```scala
val robFifo = Module(new Queue(new RobEntry, rob_entries))  // 指令队列
val robTable = Reg(Vec(rob_entries, Bool()))               // 完成状态表
val robIdCounter = RegInit(0.U(log2Up(rob_entries).W))     // ID 计数器
```

## 四、工作流程

### 指令分配流程
1. **指令入队**：解码器发送的指令进入 ROB
2. **分配 ROB ID**：为每条指令分配唯一的 ROB ID
3. **状态初始化**：在完成状态表中标记为未完成

```scala
when(io.alloc.fire) {
  robIdCounter := robIdCounter + 1.U
  robTable(robIdCounter) := false.B  // 标记为未完成
}
```

### 指令发射流程
1. **头部检查**：检查 ROB 头部指令是否未完成
2. **类型分发**：根据 bid 将指令发射到对应执行单元
3. **就绪控制**：只有目标执行单元就绪时才发射

```scala
val headEntry = robFifo.io.deq.bits
val headCompleted = robTable(headEntry.rob_id)
io.issue.valid := robFifo.io.deq.valid && !headCompleted
```

### 指令完成流程
1. **完成仲裁**：多个执行单元的完成信号通过仲裁器处理
2. **状态更新**：根据 ROB ID 更新完成状态表
3. **队列出队**：已完成的头部指令从 ROB 中移除

```scala
val completeArb = Module(new Arbiter(UInt(log2Up(rob_entries).W), 4))
when(io.complete.fire) {
  robTable(io.complete.bits) := true.B  // 标记为已完成
}
```

## 五、配置参数

### 关键配置项
- **rob_entries**: ROB 条目数量，影响乱序执行窗口大小
- **执行单元数量**: 当前支持 4 个 Ball 执行单元
- **仲裁策略**: 使用轮转仲裁处理多个完成信号

### 性能考虑
- **ROB 大小**: 更大的 ROB 支持更多乱序执行，但增加硬件开销
- **发射带宽**: 当前每周期最多发射一条指令
- **完成带宽**: 支持每周期多个指令完成

## 六、接口协议

### BallIssueInterface - 发射接口
```scala
class BallIssueInterface extends Bundle {
  val ball1 = Decoupled(new BallRsIssue)  // VecUnit 发射
  val ball2 = Decoupled(new BallRsIssue)  // BBFP 发射
  val ball3 = Decoupled(new BallRsIssue)  // im2col 发射
  val ball4 = Decoupled(new BallRsIssue)  // transpose 发射
}
```

### BallCommitInterface - 完成接口
```scala
class BallCommitInterface extends Bundle {
  val ball1 = Flipped(Decoupled(new BallRsComplete))  // VecUnit 完成
  val ball2 = Flipped(Decoupled(new BallRsComplete))  // BBFP 完成
  val ball3 = Flipped(Decoupled(new BallRsComplete))  // im2col 完成
  val ball4 = Flipped(Decoupled(new BallRsComplete))  // transpose 完成
}
```

## 七、使用示例

### 基本配置
```scala
// 在 CustomBuckyBallConfig 中配置 ROB 大小
class MyBuckyBallConfig extends CustomBuckyBallConfig {
  override val rob_entries = 16  // 16 条目 ROB
}

// 实例化保留站
val reservationStation = Module(new BallReservationStation)
```

### 连接执行单元
```scala
// 连接 VecUnit
vecUnit.io.cmd <> reservationStation.io.issue_o.ball1
reservationStation.io.commit_i.ball1 <> vecUnit.io.resp

// 连接 BBFP
bbfp.io.cmd <> reservationStation.io.issue_o.ball2
reservationStation.io.commit_i.ball2 <> bbfp.io.resp
```

## 八、调试和监控

### 状态信号
- **io.rs_rocc_o.busy**: 保留站忙碌状态
- **rob.io.empty**: ROB 空状态
- **rob.io.full**: ROB 满状态

### 性能计数器
可以添加以下性能计数器进行监控：
- 指令发射计数
- 指令完成计数
- ROB 利用率
- 各执行单元的负载分布

## 九、扩展说明

### 添加新执行单元
1. 在 `BallIssueInterface` 中添加新的发射端口
2. 在 `BallCommitInterface` 中添加对应的完成端口
3. 在保留站中添加相应的分发和仲裁逻辑
4. 更新完成信号仲裁器的端口数量

### 优化建议
- **多发射支持**: 可以扩展为每周期发射多条指令
- **动态调度**: 实现更复杂的调度算法
- **负载均衡**: 在多个同类型执行单元间进行负载均衡

## 十、相关文档

- [Ball域概览](../README.md)
- [Ball域总线](../bbus/README.md)
- [图像处理加速器](../im2col/README.md)
- [向量处理单元](../../../prototype/vector/README.md)
