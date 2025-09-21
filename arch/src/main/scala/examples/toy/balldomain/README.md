# BallDomain 球域示例实现

## 概述

该目录包含了 BuckyBall 框架中球域(BallDomain)的完整示例实现，展示了如何构建一个自定义的计算域来管理专用加速器。球域是 BuckyBall 架构中的核心概念，用于封装和管理一组相关的计算单元，提供统一的控制和数据流管理。

该目录实现了球域架构，包括：
- **BallDomain**: 球域顶层模块，管理整个计算域
- **BallController**: 球域控制器，负责指令调度和执行控制
- **DISA**: 分布式指令调度架构
- **DomainDecoder**: 域指令解码器
- **专用加速器**: 包含矩阵、向量、im2col等多种加速器实现

## 代码结构

```
balldomain/
├── BallDomain.scala      - 球域顶层模块
├── BallController.scala  - 球域控制器
├── DISA.scala           - 分布式指令调度架构
├── DomainDecoder.scala  - 域指令解码器
├── bbus/                - 球域总线系统
├── im2col/              - 图像到列转换加速器
├── matrixball/          - 矩阵运算球域
├── rs/                  - 保留站实现
└── vecball/             - 向量运算球域
```

### 文件依赖关系

**BallDomain.scala** (顶层模块)
- 集成所有子模块，提供统一的球域接口
- 管理球域内部的数据流和控制流
- 连接到系统总线和RoCC接口

**BallController.scala** (控制层)
- 实现球域的指令调度和执行控制
- 管理多个加速器之间的协调
- 提供状态管理和错误处理

**DISA.scala** (调度层)
- 分布式指令调度架构实现
- 支持多指令并发执行
- 提供动态负载均衡

**DomainDecoder.scala** (解码层)
- 球域专用指令解码
- 指令分发到相应的执行单元
- 支持复杂指令的分解和重组

## 模块说明

### BallDomain.scala

**主要功能**: 球域顶层模块，集成所有计算单元和控制逻辑

**关键组件**:

```scala
class BallDomain(implicit p: Parameters) extends LazyModule {
  val controller = LazyModule(new BallController)
  val matrixBall = LazyModule(new MatrixBall)
  val vecBall = LazyModule(new VecBall)
  val im2colUnit = LazyModule(new Im2colUnit)

  // 球域总线连接
  val bbus = LazyModule(new BBus)
  bbus.node := controller.node
  matrixBall.node := bbus.node
  vecBall.node := bbus.node
}
```

**输入输出**:
- 输入: RoCC指令接口，内存访问接口
- 输出: 计算结果，状态信息
- 边缘情况: 指令冲突处理，资源竞争管理

### BallController.scala

**主要功能**: 球域控制器，负责整个球域的运行控制

**关键组件**:

```scala
class BallController extends Module {
  val io = IO(new Bundle {
    val rocc = Flipped(new RoCCCoreIO)
    val mem = new HellaCacheIO
    val domain_ctrl = new DomainControlIO
  })

  // 指令队列和调度逻辑
  val inst_queue = Module(new Queue(new RoCCInstruction, 16))
  val scheduler = Module(new InstructionScheduler)
}
```

**调度策略**:
- 基于指令类型的静态调度
- 动态资源分配和负载均衡
- 支持指令流水线和并发执行

### DISA.scala

**主要功能**: 分布式指令调度架构

**关键组件**:

```scala
class DISA extends Module {
  val io = IO(new Bundle {
    val inst_in = Flipped(Decoupled(new Instruction))
    val exec_units = Vec(numUnits, new ExecutionUnitIO)
    val completion = Decoupled(new CompletionInfo)
  })

  // 分布式调度表
  val dispatch_table = Reg(Vec(numUnits, new DispatchEntry))
  val load_balancer = Module(new LoadBalancer)
}
```

**调度算法**:
- 轮询调度保证公平性
- 优先级调度支持关键任务
- 动态调度适应负载变化

### DomainDecoder.scala

**主要功能**: 球域指令解码器

**关键组件**:

```scala
class DomainDecoder extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(32.W))
    val decoded = Output(new DecodedInstruction)
    val valid = Output(Bool())
  })

  // 指令解码表
  val decode_table = Array(
    MATRIX_OP -> MatrixOpDecoder,
    VECTOR_OP -> VectorOpDecoder,
    IM2COL_OP -> Im2colOpDecoder
  )
}
```

**解码功能**:
- 支持多种指令格式
- 复杂指令的微码展开
- 指令依赖分析和优化

## 使用方法

### 设计特点

1. **模块化架构**: 每个加速器都是独立的模块，便于扩展和维护
2. **统一接口**: 所有加速器通过统一的球域总线进行通信
3. **灵活调度**: 支持多种调度策略，适应不同的计算模式
4. **可扩展性**: 易于添加新的加速器类型和功能

### 性能优化

1. **流水线设计**: 指令解码、调度、执行采用流水线架构
2. **并发执行**: 支持多个加速器同时工作
3. **数据管理**: 数据缓存和访问管理
4. **工作负载**: 工作负载分配

### 使用示例

```scala
// 创建球域实例
val ballDomain = LazyModule(new BallDomain)

// 连接到RoCC接口
rocc.cmd <> ballDomain.module.io.rocc.cmd
rocc.resp <> ballDomain.module.io.rocc.resp

// 配置球域参数
ballDomain.module.io.config := ballDomainConfig
```

### 注意事项

1. **资源管理**: 需要合理分配计算资源，避免资源冲突
2. **时序约束**: 注意不同模块间的时序关系和数据同步
3. **功耗控制**: 实现动态功耗管理，在不使用时关闭相应模块
4. **调试支持**: 调试接口和状态监控功能
