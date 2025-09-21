# Toy BuckyBall 示例实现

## 概述

该目录包含了 BuckyBall 框架的完整示例实现，展示了如何构建一个基于 RoCC 接口的自定义协处理器。位于 `arch/src/main/scala/examples/toy` 下，作为 BuckyBall 系统的参考实现，集成了全局解码器、Ball域和内存域。

实现的核心组件：
- **ToyBuckyBall.scala**: 主要的 RoCC 协处理器实现
- **CustomConfigs.scala**: 系统配置和 RoCC 集成配置
- **CSR.scala**: 自定义控制状态寄存器
- **balldomain/**: Ball域相关组件实现

## 代码结构

```
toy/
├── ToyBuckyBall.scala    - 主协处理器实现
├── CustomConfigs.scala   - 配置定义
├── CSR.scala            - CSR实现
└── balldomain/          - Ball域组件
```

### 文件依赖关系

**ToyBuckyBall.scala** (核心实现层)
- 继承 LazyRoCCBB，实现 RoCC 协处理器接口
- 集成 GlobalDecoder、BallDomain、MemDomain
- 管理 TileLink 连接和 DMA 组件

**CustomConfigs.scala** (配置层)
- 定义 BuckyBallCustomConfig 和 BuckyBallToyConfig
- 配置 RoCC 集成和系统参数
- 提供多核配置支持

**CSR.scala** (寄存器层)
- 实现 FenceCSR 控制寄存器
- 提供简单的 64 位寄存器接口

## 模块说明

### ToyBuckyBall.scala

**主要功能**: 实现完整的 BuckyBall RoCC 协处理器

**关键组件**:

```scala
class ToyBuckyBall(val b: CustomBuckyBallConfig)(implicit p: Parameters)
  extends LazyRoCCBB (opcodes = b.opcodes, nPTWPorts = 2) {

  val reader = LazyModule(new BBStreamReader(...))
  val writer = LazyModule(new BBStreamWriter(...))
  val xbar_node = TLXbar()
}
```

**系统架构**:
```scala
// 前端：全局解码器
val gDecoder = Module(new GlobalDecoder)

// 后端：Ball域和内存域
val ballDomain = Module(new BallDomain)
val memDomain = Module(new MemDomain)

// 响应仲裁
val respArb = Module(new Arbiter(new RoCCResponseBB()(p), 2))
```

**TileLink 连接**:
```scala
xbar_node := TLBuffer() := reader.node
xbar_node := TLBuffer() := writer.node
id_node := TLWidthWidget(b.dma_buswidth/8) := TLBuffer() := xbar_node
```

**输入输出**:
- 输入: RoCC 命令接口，PTW 接口
- 输出: RoCC 响应，TileLink 内存访问
- 边缘情况: Fence 操作时的忙等待处理

### CustomConfigs.scala

**主要功能**: 定义系统配置和 RoCC 集成

**配置类定义**:
```scala
class BuckyBallCustomConfig(
  buckyballConfig: CustomBuckyBallConfig = CustomBuckyBallConfig()
) extends Config((site, here, up) => {
  case BuildRoCCBB => up(BuildRoCCBB) ++ Seq(
    (p: Parameters) => {
      val buckyball = LazyModule(new ToyBuckyBall(buckyballConfig))
      buckyball
    }
  )
})
```

**系统配置**:
```scala
class BuckyBallToyConfig extends Config(
  new framework.rocket.WithNBuckyBallCores(1) ++
  new BuckyBallCustomConfig(CustomBuckyBallConfig()) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new WithCustomBootROM ++
  new chipyard.config.AbstractConfig
)
```

**多核支持**:
```scala
class WithMultiRoCCToyBuckyBall(harts: Int*) extends Config(...)
```

### CSR.scala

**主要功能**: 提供自定义控制状态寄存器

**实现**:
```scala
object FenceCSR {
  def apply(): UInt = RegInit(0.U(64.W))
}
```

**Fence 处理逻辑**:
```scala
val fenceCSR = FenceCSR()
val fenceSet = ballDomain.io.fence_o
val allDomainsIdle = !ballDomain.io.busy && !memDomain.io.busy

when (fenceSet) {
  fenceCSR := 1.U
  io.cmd.ready := allDomainsIdle
}
```

## 使用方法

### 系统集成

**RoCC 接口集成**:
- 通过 BuildRoCCBB 配置键注册协处理器
- 支持多核心配置
- 提供 2 个 PTW 端口用于地址转换

**域间通信**:
```scala
// BallDomain -> MemDomain 桥接
ballDomain.io.sramRead <> memDomain.io.ballDomain.sramRead
ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite
```

**DMA 连接**:
```scala
memDomain.io.dma.read.req <> outer.reader.module.io.req
memDomain.io.dma.write.req <> outer.writer.module.io.req
```

### 注意事项

1. **Fence 语义**: 使用 CSR 实现 Fence 操作的同步
2. **忙等待检测**: 防止仿真长时间停顿的断言检查
3. **TLB 集成**: TLB 功能集成在 MemDomain 内部
4. **响应仲裁**: BallDomain 优先级高于 MemDomain
5. **配置依赖**: 需要正确配置 CustomBuckyBallConfig 参数
