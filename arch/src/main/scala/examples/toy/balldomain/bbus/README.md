# BBus 球域总线系统

## 概述

该目录包含了 BuckyBall 球域总线系统的实现，主要负责管理球域内多个Ball节点对SRAM资源的访问。总线系统基于 framework.blink 中的 BBusNode 实现，提供了SRAM资源的仲裁和路由功能。

该目录实现了两个核心组件：
- **BallBus**: 球域总线主模块，管理多个Ball节点的SRAM访问
- **BBusRouter**: 总线路由器，提供Blink接口的路由功能

## 代码结构

```
bbus/
├── BallBus.scala    - 球域总线主模块
└── router.scala     - 总线路由器实现
```

### 文件依赖关系

**BallBus.scala** (主模块)
- 创建多个BBusNode实例来管理Ball节点
- 连接外部SRAM接口到各个Ball节点
- 实现SRAM资源的分配和仲裁

**router.scala** (路由模块)
- 基于BBusNode实现路由功能
- 提供Blink协议的接口封装

## 模块说明

### BallBus.scala

**主要功能**: 球域总线主模块，管理多个Ball节点对SRAM资源的访问

**关键组件**:

```scala
class BallBus(maxReadBW: Int, maxWriteBW: Int, numBalls: Int) extends LazyModule {
  // 创建多个BBusNode
  val ballNodes = Seq.fill(numBalls) {
    new BBusNode(BallParams(sramReadBW = maxReadBW, sramWriteBW = maxWriteBW))
  }

  // 外部SRAM接口
  val io = IO(new Bundle {
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(...)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(...)))
  })
}
```

**资源分配策略**:
- 前 `sp_banks` 个端口连接到scratchpad SRAM
- 接下来 `acc_banks` 个端口连接到accumulator SRAM
- 多余的端口设置为无效状态
- 所有Ball节点共享相同的SRAM资源

**输入输出**:
- 输入: 来自各Ball节点的SRAM访问请求
- 输出: 连接到外部SRAM的读写接口
- 边缘情况: 处理超出配置范围的端口，设置为DontCare

**依赖项**: framework.blink.BBusNode, framework.builtin.memdomain.mem

### router.scala

**主要功能**: 总线路由器，提供Blink协议接口的路由功能

**关键组件**:

```scala
class BBusRouter extends LazyModule {
  val node = new BBusNode(BallParams(
    sramReadBW = b.sp_banks,
    sramWriteBW = b.sp_banks
  ))

  val io = IO(new Bundle {
    val blink = Flipped(new BlinkBundle(node.edges.in.head))
  })
}
```

**路由功能**:
- 基于BBusNode实现标准的Ball节点接口
- 提供Blink协议的封装和转换
- 支持配置化的读写带宽参数

**输入输出**:
- 输入: Blink协议接口
- 输出: BBusNode标准接口
- 边缘情况: 依赖node.edges.in.head的有效性

**依赖项**: framework.blink.BlinkBundle, framework.blink.BBusNode

## 使用方法

### 配置参数

总线系统的配置通过以下参数控制：
- `maxReadBW`: 最大读带宽（端口数量）
- `maxWriteBW`: 最大写带宽（端口数量）
- `numBalls`: Ball节点数量
- `b.sp_banks`: Scratchpad Bank数量
- `b.acc_banks`: Accumulator Bank数量

### 资源管理

1. **SRAM端口分配**: 按照scratchpad优先、accumulator次之的顺序分配端口
2. **多Ball共享**: 所有Ball节点共享相同的SRAM资源池
3. **端口复用**: 超出配置的端口被设置为无效状态以节省资源

### 使用示例

```scala
// 创建球域总线
val ballBus = LazyModule(new BallBus(
  maxReadBW = 8,
  maxWriteBW = 8,
  numBalls = 4
))

// 连接外部SRAM
scratchpad.io.read <> ballBus.module.io.sramRead
scratchpad.io.write <> ballBus.module.io.sramWrite
accumulator.io.read <> ballBus.module.io.accRead
accumulator.io.write <> ballBus.module.io.accWrite
```

### 注意事项

1. **资源冲突**: 多个Ball节点可能同时访问相同的SRAM资源，需要上层协调
2. **带宽限制**: 实际可用带宽受限于配置的最大读写带宽参数
3. **端口映射**: 确保SRAM端口数量与配置参数匹配，避免越界访问
4. **时序约束**: BBusNode的时序要求需要与外部SRAM接口匹配
