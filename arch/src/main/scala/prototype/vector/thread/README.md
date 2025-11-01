# Thread Module

## Overview

The thread module implements thread abstractions in the vector processing unit, located at `prototype/vector/thread`. This module defines the basic structure and specific implementations of threads, constructing threads with specific functionality by combining different operations (Op) and bindings (Bond).

## File Structure

```
thread/
├── BaseThread.scala    - Thread base class definition
├── CasThread.scala     - Cascade operation thread
└── MulThread.scala     - Multiplication operation thread
```

## Core Components

### BaseThread - Thread Base Class

BaseThread is the base class for all threads, defining basic thread parameters and configuration:

```scala
class BaseThread(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {})
  val params = p
  val threadMap = p(ThreadMapKey)
  val threadParam = threadMap.getOrElse(
    p(ThreadKey).get.threadName,
    throw new Exception(s"ThreadParam not found for threadName: ${p(ThreadKey).get.threadName}")
  )
  val opParam = p(ThreadOpKey).get
  val bondParam = p(ThreadBondKey).get
}
```

### Parameter Definition

The thread module uses the following parameter structure:

```scala
case class ThreadParam(lane: Int, attr: String, threadName: String, Op: OpParam)
case class OpParam(OpType: String, bondType: BondParam)
case class BondParam(bondType: String, inputWidth: Int = 8, outputWidth: Int = 32)
```

Parameter description:
- `lane`: Vector lane count
- `threadName`: Thread name identifier
- `OpType`: Operation type ("cascade", "mul")
- `bondType`: Binding type ("vvv")
- `inputWidth`: Input data width, default 8 bits
- `outputWidth`: Output data width, default 32 bits

## Specific Thread Implementations

### CasThread - Cascade Operation Thread

CasThread implements cascade addition operation, combining CascadeOp and VVVBond:

```scala
class CasThread(implicit p: Parameters) extends BaseThread
  with CanHaveCascadeOp
  with CanHaveVVVBond {

  // Connect CascadeOp and VVVBond
  for {
    op <- cascadeOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

Function: Performs element-wise addition operation on two input vectors.

### MulThread - Multiplication Operation Thread

MulThread implements multiplication operation, combining MulOp and VVVBond:

```scala
class MulThread(implicit p: Parameters) extends BaseThread
  with CanHaveMulOp
  with CanHaveVVVBond {

  // Connect MulOp and VVVBond
  for {
    op <- mulOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

Function: Implements vector multiplication operation, supporting per-cycle result output.

## Configuration System

The thread module uses Chipyard's configuration system for parameterization:

```scala
case object ThreadKey extends Field[Option[ThreadParam]](None)
case object ThreadOpKey extends Field[Option[OpParam]](None)
case object ThreadBondKey extends Field[Option[BondParam]](None)
case object ThreadMapKey extends Field[Map[String, ThreadParam]](Map.empty)
```

Configuration key description:
- `ThreadKey`: Current thread parameter
- `ThreadOpKey`: Operation parameter
- `ThreadBondKey`: Binding parameter
- `ThreadMapKey`: Thread mapping table

## Usage

### Creating Thread Instance

```scala
// Configure parameters
val threadParam = ThreadParam(
  lane = 4,
  attr = "vector",
  threadName = "mul_thread",
  Op = OpParam("mul", BondParam("vvv", 8, 32))
)

// Create thread
val mulThread = Module(new MulThread()(
  new Config((site, here, up) => {
    case ThreadKey => Some(threadParam)
    case ThreadOpKey => Some(threadParam.Op)
    case ThreadBondKey => Some(threadParam.Op.bondType)
  })
))
```

### Connecting Interfaces

Threads interact data through VVV binding interface:

```scala
// Input data
mulThread.io.in.valid := inputValid
mulThread.io.in.bits.in1 := inputVector1
mulThread.io.in.bits.in2 := inputVector2

// Output data
outputValid := mulThread.io.out.valid
outputVector := mulThread.io.out.bits.out
mulThread.io.out.ready := outputReady
```

## Related Modules

- [Vector Operation Module](../op/README.md) - Provides specific computation operations
- [Binding Module](../bond/README.md) - Provides data interfaces and synchronization mechanisms
- [Vector Processing Unit](../README.md) - Upper-level vector processor
