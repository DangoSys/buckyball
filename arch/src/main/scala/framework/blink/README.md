# Blink Interconnect System

## Overview

This directory implements the Buckyball Blink interconnect system, providing connection negotiation between Ball and BBus based on the Diplomacy framework. Located in `arch/src/main/scala/framework/blink`, it serves as the system interconnect layer, managing SRAM bandwidth resource allocation and negotiation.

Core components:
- **ball.scala**: Ball module as Diplomacy source
- **bbus.scala**: BBus module as Diplomacy sink
- **blink.scala**: Blink protocol definition and nexus node

## Code Structure

```
blink/
├── ball.scala    - Ball module (source)
├── bbus.scala    - BBus module (sink)
└── blink.scala   - Blink protocol and nexus
```

Ball nodes request bandwidth requirements, BBus nodes provide system bandwidth capabilities, and they negotiate at the BlinkNode.

Ball nodes -> BlinkNode (nexus) -> BBusNode (sink)

### File Dependencies

**blink.scala** (Protocol definition layer)
- Defines BBusParams, BallParams, BlinkParams
- Implements BlinkNodeImp and various Node types
- Provides BlinkBundle interface definition

**ball.scala** (source side)
- Extends LazyModule, uses BallNode
- Sends bandwidth requirement parameters upstream
- Provides default interface implementation

**bbus.scala** (sink side)
- Extends LazyModule, uses BBusNode
- Receives connection requests from Ball
- Implements bandwidth resource allocation

## Module Description

### blink.scala

**Main functionality**: Defines Blink protocol parameter types and Diplomacy nodes

**Parameter definition**:
```scala
case class BBusParams (sramReadBW: Int = 2, sramWriteBW: Int = 1)  // DownParam
case class BlinkParams(sramReadBW: Int = 2, sramWriteBW: Int = 1)  // EdgeParam
case class BallParams (sramReadBW: Int = 2, sramWriteBW: Int = 1)  // UpParam
```

**Protocol interface**:
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

**Bandwidth negotiation logic**:
```scala
def edge(pd: BBusParams, pu: BallParams, p: Parameters, sourceInfo: SourceInfo) = {
  require(pd.sramReadBW >= pu.sramReadBW, "BBus read bandwidth must be >= Ball requirement")
  require(pd.sramWriteBW >= pu.sramWriteBW, "BBus write bandwidth must be >= Ball requirement")
  BlinkParams(pd.sramReadBW, pd.sramWriteBW)
}
```

### ball.scala

**Main functionality**: Ball module as Diplomacy source, sends bandwidth requirements upstream

**Key implementation**:
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

**Default interface behavior**:
```scala
// Ball interface default values (overridden by specific Ball implementations)
io.cmd.req.ready := false.B
io.cmd.resp.valid := false.B
io.cmd.resp.bits := DontCare
```

### bbus.scala

**Main functionality**: BBus module as Diplomacy sink, receives Ball connections

**Key implementation**:
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

**Interface specification**:
```
Frontend input: RS cmd, MemDomain bridge: n0 * sramRead(), m0 * sramWrite()
Backend output: Ball cmd, n1 * sramRead(), m1 * sramWrite()
Requirement: n0 >= n1, m0 >= m1
```

## Usage

### Usage Examples

**Create Ball and BBus connection**:
```scala
val ball = LazyModule(new Ball(BallParams(sramReadBW = 2, sramWriteBW = 1)))
val bbus = LazyModule(new BBus(BallParams(sramReadBW = 4, sramWriteBW = 2)))
bbus.node := ball.node
```

**Blink nexus usage**:
```scala
val blink = LazyModule(new Blink())
// Connect multiple Ball and BBus
```

### Notes

1. **Bandwidth constraints**: BBus bandwidth must be >= connected Ball bandwidth requirements
2. **Diplomacy negotiation**: Parameters negotiated at compile time through Diplomacy framework
3. **Interface connection**: Use `<>` operator to connect Diplomacy-generated interfaces
4. **Default implementation**: Ball and BBus provide default interface behavior, needs to be overridden by specific implementations
