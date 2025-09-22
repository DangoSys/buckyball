# Blink 互连系统实现

## 概述

该目录实现了 BuckyBall 的 Blink 互连系统，基于 Diplomacy 框架提供 Ball 和 BBus 之间的连接协商。位于 `arch/src/main/scala/framework/blink` 下，作为系统互连层，负责管理 SRAM 带宽资源的分配和协商。

实现的核心组件：
- **ball.scala**: Ball 模块，作为 Diplomacy source 端
- **bbus.scala**: BBus 模块，作为 Diplomacy sink 端
- **blink.scala**: Blink 协议定义和 nexus 节点

## 代码结构

```
blink/
├── ball.scala    - Ball 模块(source)
├── bbus.scala    - BBus 模块(sink)
└── blink.scala   - Blink 协议和 nexus
```

ball节点请求带宽需求
bbus节点给出系统的带宽能力
两者在blinkNode协商

Ball nodes -> BlinkNode (nexus) -> BBusNode (sink)

### 文件依赖关系

**blink.scala** (协议定义层)
- 定义 BBusParams, BallParams, BlinkParams
- 实现 BlinkNodeImp 和各种 Node 类型
- 提供 BlinkBundle 接口定义

**ball.scala** (source 端)
- 继承 LazyModule，使用 BallNode
- 向上发送带宽需求参数
- 提供默认的接口实现

**bbus.scala** (sink 端)
- 继承 LazyModule，使用 BBusNode
- 接收来自 Ball 的连接请求
- 实现带宽资源的分配

## 模块说明

### blink.scala

**主要功能**: 定义 Blink 协议的参数类型和 Diplomacy 节点

**参数定义**:
```scala
case class BBusParams (sramReadBW: Int = 2, sramWriteBW: Int = 1)  // DownParam
case class BlinkParams(sramReadBW: Int = 2, sramWriteBW: Int = 1)  // EdgeParam
case class BallParams (sramReadBW: Int = 2, sramWriteBW: Int = 1)  // UpParam
```

**协议接口**:
```scala
class BlinkBundle(params: BlinkParams) extends Bundle {
  val cmd = new Bundle {
    val req = Flipped(Decoupled(new BallRsIssue))
    val resp = Decoupled(new BallRsComplete)
  }
  val data = new Bundle {
    val sramRead = Vec(params.sramReadBW, Flipped(new SramReadIO(...)))
    val sramWrite = Vec(params.sramWriteBW, Flipped(new SramWriteIO(...)))
  }
  val status = Decoupled(new BlinkStatus())
}
```

**带宽协商逻辑**:
```scala
def edge(pd: BBusParams, pu: BallParams, p: Parameters, sourceInfo: SourceInfo) = {
  require(pd.sramReadBW >= pu.sramReadBW, "BBus 读带宽必须大于等于 Ball 需求")
  require(pd.sramWriteBW >= pu.sramWriteBW, "BBus 写带宽必须大于等于 Ball 需求")
  BlinkParams(pd.sramReadBW, pd.sramWriteBW)
}
```

### ball.scala

**主要功能**: Ball 模块作为 Diplomacy source 端，向上发送带宽需求

**关键实现**:
```scala
class Ball(params: BallParams) extends LazyModule {
  val node = new BallNode(Seq(BBusParams(params.sramReadBW, params.sramWriteBW)))

  lazy val module = new LazyModuleImp(this) {
    val edgeParams = node.edges.out.head
    val io = IO(new BlinkBundle(edgeParams))
    node.out.head._1 <> io
  }
}
```

**默认接口行为**:
```scala
// Ball接口默认值（由各个具体Ball实现来覆盖）
io.cmd.req.ready := false.B
io.cmd.resp.valid := false.B
io.cmd.resp.bits := DontCare
```

### bbus.scala

**主要功能**: BBus 模块作为 Diplomacy sink 端，接收 Ball 的连接

**关键实现**:
```scala
class BBus(params: BallParams) extends LazyModule {
  val node = new BBusNode(params)

  lazy val module = new LazyModuleImp(this) {
    val edgeParams = node.edges.in.head
    val io = IO(new Bundle {
      val blink = Flipped(new BlinkBundle(edgeParams))
    })
    node.in.head._1 <> io.blink
  }
}
```

**接口规范**:
```
前端输入: RS cmd, MemDomain bridge: n0 * sramRead(), m0 * sramWrite()
后端输出: Ball cmd, n1 * sramRead(), m1 * sramWrite()
要求: n0 >= n1, m0 >= m1
```

## 使用方法

### 使用方法

**创建 Ball 和 BBus 连接**:
```scala
val ball = LazyModule(new Ball(BallParams(sramReadBW = 2, sramWriteBW = 1)))
val bbus = LazyModule(new BBus(BallParams(sramReadBW = 4, sramWriteBW = 2)))
bbus.node := ball.node
```

**Blink nexus 使用**:
```scala
val blink = LazyModule(new Blink())
// 连接多个 Ball 和 BBus
```

### 注意事项

1. **带宽约束**: BBus 的带宽必须大于等于连接的 Ball 带宽需求
2. **Diplomacy 协商**: 参数在编译时通过 Diplomacy 框架协商确定
3. **接口连接**: 使用 `<>` 操作符连接 Diplomacy 生成的接口
4. **默认实现**: Ball 和 BBus 提供默认接口行为，需要具体实现覆盖
