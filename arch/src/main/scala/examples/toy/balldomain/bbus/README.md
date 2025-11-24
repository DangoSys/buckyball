# BBus Ball Domain Bus System

## Overview

This directory contains the implementation of Buckyball's ball domain bus system, primarily responsible for managing SRAM resource access by multiple Ball nodes within the ball domain. The bus system is implemented based on BBusNode from framework.blink, providing SRAM resource arbitration and routing functionality.

This directory implements two core components:
- **BallBus**: Ball domain bus main module, manages SRAM access by multiple Ball nodes
- **BBusRouter**: Bus router, provides routing functionality for Blink interface

## Code Structure

```
bbus/
├── BallBus.scala    - Ball domain bus main module
└── router.scala     - Bus router implementation
```

### File Dependencies

**BallBus.scala** (Main module)
- Creates multiple BBusNode instances to manage Ball nodes
- Connects external SRAM interfaces to each Ball node
- Implements SRAM resource allocation and arbitration

**router.scala** (Routing module)
- Implements routing functionality based on BBusNode
- Provides Blink protocol interface encapsulation

## Module Description

### BallBus.scala

**Main functionality**: Ball domain bus main module, manages SRAM resource access by multiple Ball nodes

**Key components**:

```scala
class BallBus(maxReadBW: Int, maxWriteBW: Int, numBalls: Int) extends LazyModule {
  // Create multiple BBusNode instances
  val ballNodes = Seq.fill(numBalls) {
    new BBusNode(BallParams(sramReadBW = maxReadBW, sramWriteBW = maxWriteBW))
  }

  // External SRAM interfaces
  val io = IO(new Bundle {
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(...)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(...)))
  })
}
```

**Resource allocation strategy**:
- First `sp_banks` ports connected to scratchpad SRAM
- Next `acc_banks` ports connected to accumulator SRAM
- Excess ports set to invalid state
- All Ball nodes share the same SRAM resources

**Inputs/Outputs**:
- Input: SRAM access requests from each Ball node
- Output: Read/write interfaces connected to external SRAM
- Edge cases: Handle ports beyond configuration range, set to DontCare

**Dependencies**: framework.blink.BBusNode, framework.builtin.memdomain.mem

### router.scala

**Main functionality**: Bus router, provides routing functionality for Blink protocol interface

**Key components**:

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

**Routing functionality**:
- Implements standard Ball node interface based on BBusNode
- Provides Blink protocol encapsulation and conversion
- Supports configurable read/write bandwidth parameters

**Inputs/Outputs**:
- Input: Blink protocol interface
- Output: BBusNode standard interface
- Edge cases: Depends on validity of node.edges.in.head

**Dependencies**: framework.blink.BlinkBundle, framework.blink.BBusNode

## Usage

### Configuration Parameters

Bus system configuration is controlled by the following parameters:
- `maxReadBW`: Maximum read bandwidth (port count)
- `maxWriteBW`: Maximum write bandwidth (port count)
- `numBalls`: Ball node count
- `b.sp_banks`: Scratchpad bank count
- `b.acc_banks`: Accumulator bank count

### Resource Management

1. **SRAM port allocation**: Allocate ports in order of scratchpad first, accumulator second
2. **Multi-Ball sharing**: All Ball nodes share the same SRAM resource pool
3. **Port reuse**: Ports beyond configuration are set to invalid state to save resources

### Usage Example

```scala
// Create ball domain bus
val ballBus = LazyModule(new BallBus(
  maxReadBW = 8,
  maxWriteBW = 8,
  numBalls = 4
))

// Connect external SRAM
scratchpad.io.read <> ballBus.module.io.sramRead
scratchpad.io.write <> ballBus.module.io.sramWrite
accumulator.io.read <> ballBus.module.io.accRead
accumulator.io.write <> ballBus.module.io.accWrite
```

### Notes

1. **Resource conflicts**: Multiple Ball nodes may access the same SRAM resources simultaneously, requiring upper-level coordination
2. **Bandwidth limitations**: Actual available bandwidth is limited by configured maximum read/write bandwidth parameters
3. **Port mapping**: Ensure SRAM port count matches configuration parameters to avoid out-of-bounds access
4. **Timing constraints**: BBusNode timing requirements need to match external SRAM interfaces
