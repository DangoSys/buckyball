# Ball域保留站 (Ball Reservation Station)

## 核心功能

通用的Ball域保留站实现，支持：
- ✅ 环状ROB（循环队列）
- ✅ 顺序发射，最多发射ROB深度一半的指令
- ✅ 乱序完成和乱序提交
- ✅ 可配置的顺序/乱序响应模式
- ✅ 动态Ball设备数量支持

## 文件结构

```
rs/
├── reservationStation.scala  - 保留站主模块，连接ROB和Ball设备
├── rob.scala                 - 环状ROB实现
└── README.md                 - 本文档
```

## 架构设计

### 整体流水线

```
Ball Decoder → ROB分配 → ROB发射 → Ball设备执行 → 完成信号 → ROB提交
     ↓                                                        ↑
  立即响应(可配置)                                  顺序/乱序过滤(可配置)
```

### 模块职责

**保留站 (BallReservationStation)**：
- 接收Ball域解码指令
- 管理多个Ball设备的发射和完成
- 根据配置决定响应策略（顺序/乱序）
- 在顺序模式下过滤非head的完成信号

**ROB (Reorder Buffer)**：
- 环状队列管理（循环的rob_id）
- 顺序发射，限制inflight数量
- 乱序提交（每周期提交所有已完成指令）
- 暴露内部状态供保留站决策

## ROB实现细节

### 环状队列结构

```scala
// 核心状态
val robEntries   = Reg(Vec(b.rob_entries, new RobEntry))  // 指令存储
val robValid     = Reg(Vec(b.rob_entries, Bool()))         // 条目有效
val robIssued    = Reg(Vec(b.rob_entries, Bool()))         // 已发射
val robComplete  = Reg(Vec(b.rob_entries, Bool()))         // 已完成

// 循环队列指针
val headPtr      = Reg(UInt())  // 最老的未提交指令
val tailPtr      = Reg(UInt())  // 下一个分配位置
val robIdCounter = Reg(UInt())  // ROB ID循环计数器（0 ~ rob_entries-1）

// 发射限制
val issuedCount  = Reg(UInt())  // 已发射未完成的指令数
val maxIssueLimit = (b.rob_entries / 2).U  // 最多发射一半
```

### ROB ID循环

```scala
// 分配时循环递增
when(io.alloc.fire) {
  robIdCounter := Mux(robIdCounter === (b.rob_entries - 1).U,
                      0.U, robIdCounter + 1.U)
}
```

### 顺序发射逻辑

从head指针开始扫描，找到第一个未发射的指令：

```scala
// 扫描所有位置
for (i <- 0 until b.rob_entries) {
  val ptr = (headPtr + i.U) % b.rob_entries.U
  scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
}

// 优先级编码器找到第一个
val issuePtr = PriorityEncoder(scanValid)

// 检查发射限制
val canIssue = scanValid.orR && (issuedCount < maxIssueLimit)
```

### 乱序提交逻辑

每周期提交所有已完成的指令，然后更新head指针：

```scala
// 提交所有完成的指令
for (i <- 0 until b.rob_entries) {
  when(robValid(i.U) && robComplete(i.U)) {
    robValid(i.U) := false.B
    robIssued(i.U) := false.B
    robComplete(i.U) := false.B
  }
}

// head指针跳过所有已提交的位置，移动到第一个有效未完成的指令
```

### 暴露的状态信号

```scala
io.empty          // ROB是否为空
io.full           // ROB是否已满
io.head_ptr       // 头指针位置
io.issued_count   // inflight指令数
io.entry_valid    // 每个条目是否有效
io.entry_complete // 每个条目是否完成
```

## 保留站实现细节

### 发射逻辑

根据指令的`bid`（Ball ID）分发到对应Ball设备：

```scala
for (i <- 0 until numBalls) {
  val ballId = BallRsRegists(i).ballId.U
  io.issue_o.balls(i).valid := rob.io.issue.valid &&
                               rob.io.issue.bits.cmd.bid === ballId
  io.issue_o.balls(i).bits  := rob.io.issue.bits
}

// ROB ready：只有目标Ball设备ready时才能发射
rob.io.issue.ready := VecInit(
  BallRsRegists.zipWithIndex.map { case (info, idx) =>
    (rob.io.issue.bits.cmd.bid === info.ballId.U) &&
    io.issue_o.balls(idx).ready
  }
).asUInt.orR
```

### 完成信号处理（关键）

**乱序模式**：接受所有完成信号

```scala
if (b.rs_out_of_order_response) {
  rob.io.complete <> completeArb.io.out
}
```

**顺序模式**：只接受`rob_id == head_ptr`的完成信号

```scala
else {
  val isHeadComplete = completeArb.io.out.bits === rob.io.head_ptr
  rob.io.complete.valid := completeArb.io.out.valid && isHeadComplete
  rob.io.complete.bits  := completeArb.io.out.bits
  // 非head指令会被阻塞等待
  completeArb.io.out.ready := rob.io.complete.ready && isHeadComplete
}
```

## 配置参数

### BaseConfigs.scala

```scala
case class CustomBuckyBallConfig(
  rob_entries: Int = 16,                      // ROB条目数量
  rs_out_of_order_response: Boolean = true,   // 乱序响应模式
  ...
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rob_entries` | 16 | ROB深度，影响乱序窗口大小 |
| `rs_out_of_order_response` | true | true=乱序响应，false=顺序响应 |

## 使用示例

### 注册Ball设备

```scala
val ballDevices = Seq(
  BallRsRegist(ballId = 0, ballName = "VectorUnit"),
  BallRsRegist(ballId = 1, ballName = "MatrixUnit"),
  BallRsRegist(ballId = 2, ballName = "LoadUnit"),
  BallRsRegist(ballId = 3, ballName = "StoreUnit")
)

val rs = Module(new BallReservationStation(ballDevices))
```

### 连接接口

```scala
// 输入：来自Ball域解码器
rs.io.ball_decode_cmd_i <> decoder.io.ball_cmd_o

// 输出：到各个Ball设备
rs.io.issue_o.balls(0) <> vectorUnit.io.cmd_i
rs.io.issue_o.balls(1) <> matrixUnit.io.cmd_i
rs.io.issue_o.balls(2) <> loadUnit.io.cmd_i
rs.io.issue_o.balls(3) <> storeUnit.io.cmd_i

// 完成信号：从各个Ball设备
rs.io.commit_i.balls(0) <> vectorUnit.io.complete_o
rs.io.commit_i.balls(1) <> matrixUnit.io.complete_o
rs.io.commit_i.balls(2) <> loadUnit.io.complete_o
rs.io.commit_i.balls(3) <> storeUnit.io.complete_o

// RoCC响应
rocc.resp <> rs.io.rs_rocc_o.resp
rocc.busy := rs.io.rs_rocc_o.busy
```

## 性能特性

### 乱序模式 (rs_out_of_order_response = true)

**优点**：
- ✅ 高吞吐量
- ✅ Ball设备不会被阻塞
- ✅ 充分利用ROB容量
- ✅ 适合高性能场景

**缺点**：
- ❌ 不保证严格的指令顺序
- ❌ 调试困难

**适用场景**：
- 独立的Ball计算任务
- 无数据依赖的批量操作
- 追求最大吞吐量

### 顺序模式 (rs_out_of_order_response = false)

**优点**：
- ✅ 严格按程序顺序提交
- ✅ 行为可预测
- ✅ 便于调试
- ✅ 支持精确异常

**缺点**：
- ❌ 吞吐量较低
- ❌ Ball设备可能被阻塞（等待head完成）
- ❌ ROB利用率可能较低

**适用场景**：
- 有数据依赖的操作序列
- 需要调试和验证
- 对顺序有严格要求

### 发射限制的影响

```
最大inflight数 = rob_entries / 2
```

**ROB=16时**：
- 最多同时发射8条指令
- 剩余8个位置用于缓冲新指令
- 平衡发射压力和缓冲能力

## 性能调优建议

### 增加ROB深度

```scala
override val rob_entries = 32  // 增加到32
```
- ✅ 更大的乱序窗口
- ✅ 更多指令可并行执行
- ❌ 面积和功耗增加

### 调整发射限制

如果需要修改发射比例，编辑`rob.scala`：

```scala
val maxIssueLimit = (b.rob_entries * 3 / 4).U  // 改为3/4
```

### 混合模式（未来扩展）

保留站可以利用`rob.io.entry_complete`等信号实现更复杂的策略：

```scala
// 示例：根据完成情况动态调整
val completedRatio = PopCount(rob.io.entry_complete) / rob_entries.U
val allowResponse = (completedRatio > threshold.U) || rob.io.empty
```

## 时序图

### 乱序模式执行流程

```
周期 | 动作              | headPtr | issuedCount | ROB状态
-----|-------------------|---------|-------------|------------------
1    | 分配指令0         | 0       | 0           | [0:未发射]
2    | 发射指令0         | 0       | 1           | [0:已发射]
3    | 分配指令1，发射1  | 0       | 2           | [0:已发射, 1:已发射]
4    | 指令1完成         | 0       | 1           | [0:已发射, 1:完成]
5    | 指令1提交         | 0       | 1           | [0:已发射, 1:空]
6    | 指令0完成         | 0       | 0           | [0:完成, 1:空]
7    | 指令0提交         | 2       | 0           | [0:空, 1:空]
```

### 顺序模式执行流程

```
周期 | 动作              | headPtr | 完成信号        | 提交
-----|-------------------|---------|-----------------|----------
1    | 分配指令0         | 0       | -               | -
2    | 发射指令0         | 0       | -               | -
3    | 分配指令1，发射1  | 0       | -               | -
4    | 指令1完成         | 0       | rob_id=1 ❌阻塞 | -
5    | 指令1继续等待     | 0       | rob_id=1 ❌阻塞 | -
6    | 指令0完成         | 0       | rob_id=0 ✅接受 | 指令0
7    | head移动          | 1       | -               | -
8    | 指令1重新尝试     | 1       | rob_id=1 ✅接受 | 指令1
```

## 调试技巧

### 查看ROB状态

```scala
when(rob.io.alloc.fire) {
  printf("Alloc: rob_id=%d, bid=%d\n",
    rob.io.alloc.bits.rob_id, rob.io.alloc.bits.cmd.bid)
}

when(rob.io.issue.fire) {
  printf("Issue: rob_id=%d, head=%d, issued_count=%d\n",
    rob.io.issue.bits.rob_id, rob.io.head_ptr, rob.io.issued_count)
}

when(rob.io.complete.fire) {
  printf("Complete: rob_id=%d, head=%d\n",
    rob.io.complete.bits, rob.io.head_ptr)
}
```

### 常见问题排查

**问题1：ROB一直满**
- 检查Ball设备是否正常完成
- 检查完成信号是否正确连接
- 顺序模式下检查是否head指令卡住

**问题2：指令没有发射**
- 检查`issued_count`是否达到上限
- 检查Ball设备的ready信号
- 检查bid是否匹配注册的Ball设备

**问题3：顺序模式下性能低**
- 考虑切换到乱序模式
- 增加ROB深度
- 优化Ball设备执行延迟

## 相关文档

- [框架概览](../../../README.md)
- [Ball域实现示例](../../../../examples/toy/balldomain/)
- [BaseConfigs配置说明](../../BaseConfigs.scala)

## 设计权衡

| 设计选择 | 原因 |
|---------|------|
| ROB固定乱序提交 | 简化ROB逻辑，提高性能 |
| 保留站控制顺序/乱序 | 策略灵活，易于扩展 |
| 发射限制=深度/2 | 平衡并行度和缓冲能力 |
| 暴露ROB内部状态 | 支持复杂的调度策略 |
| 环状队列 | ROB ID可循环使用，支持长时间运行 |
