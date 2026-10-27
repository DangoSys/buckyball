# Binding Module

## Overview

The binding module implements data interfaces and synchronization mechanisms in the vector processing unit, located at `prototype/vector/bond`. This module defines inter-thread data transfer interfaces, supporting different types of data binding patterns.

## File Structure

```
bond/
├── BondWrapper.scala    - Binding wrapper base class
└── vvv.scala           - VVV binding implementation
```

## Core Components

### VVV - Vector-to-Vector Binding

VVV (Vector-Vector-Vector) binding implements a data interface from dual input vectors to single output vector:

```scala
class VVV(implicit p: Parameters) extends Bundle {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val inputWidth = bondParam.inputWidth
  val outputWidth = bondParam.outputWidth

  // Input interface (Flipped Decoupled)
  val in = Flipped(Decoupled(new Bundle {
    val in1 = Vec(lane, UInt(inputWidth.W))
    val in2 = Vec(lane, UInt(inputWidth.W))
  }))

  // Decoupled output interface
  val out = Decoupled(new Bundle {
    val out = Vec(lane, UInt(outputWidth.W))
  })
}
```

#### Interface Description

**Input interface**:
- `in.bits.in1`: First input vector, width is `inputWidth`
- `in.bits.in2`: Second input vector, width is `inputWidth`
- `in.valid`: Input data valid signal
- `in.ready`: Input ready signal

**Output interface**:
- `out.bits.out`: Output vector, width is `outputWidth`
- `out.valid`: Output data valid signal
- `out.ready`: Output ready signal

#### Parameter Configuration

VVV binding parameters are obtained through the configuration system:

```scala
val lane = p(ThreadKey).get.lane                    // Vector lane count
val bondParam = p(ThreadBondKey).get                // Binding parameter
val inputWidth = bondParam.inputWidth               // Input width
val outputWidth = bondParam.outputWidth             // Output width
```

### CanHaveVVVBond - VVV Binding Trait

The CanHaveVVVBond trait provides VVV binding functionality for threads:

```scala
trait CanHaveVVVBond { this: BaseThread =>
  val vvvBond = params(ThreadBondKey).filter(_.bondType == "vvv").map { bondParam =>
    IO(new VVV()(params))
  }

  def getVVVBond = vvvBond
}
```

#### Usage

Thread classes gain VVV binding capability by mixing in this trait:

```scala
class MulThread(implicit p: Parameters) extends BaseThread
  with CanHaveMulOp
  with CanHaveVVVBond {

  // Connect operation and binding
  for {
    op <- mulOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

### BondWrapper - Binding Wrapper

BondWrapper provides Diplomacy-based binding encapsulation:

```scala
abstract class BondWrapper(implicit p: Parameters) extends LazyModule {
  val bondName = "vvv"

  def to[T](name: String)(body: => T): T = {
    LazyScope(s"bond_to_${name}", s"Bond_${bondName}_to_${name}") { body }
  }

  def from[T](name: String)(body: => T): T = {
    LazyScope(s"bond_from_${name}", s"Bond_${bondName}_from_${name}") { body }
  }
}
```

#### Scope Management

BondWrapper provides named scope management functionality:
- `to()`: Creates binding scope in output direction
- `from()`: Creates binding scope in input direction

## Binding Types

### VVV Binding Pattern

VVV binding supports the following data flow patterns:

1. **Dual input single output**: Two vector inputs, one vector output
2. **Width conversion**: Supports different input and output widths
3. **Vector parallelism**: Supports multi-lane parallel data transmission

### Data Flow Control

VVV binding uses Decoupled interface for flow control:

```scala
// Producer side
producer.io.out.valid := dataReady
producer.io.out.bits.in1 := inputVector1
producer.io.out.bits.in2 := inputVector2

// Consumer side
consumer.io.in.ready := canAcceptData
when(consumer.io.in.fire) {
  processData(consumer.io.in.bits.out)
}
```

## Configuration Parameters

### Binding Parameters

Binding parameters are defined through `BondParam`:

```scala
case class BondParam(
  bondType: String,           // Binding type ("vvv")
  inputWidth: Int = 8,        // Input width
  outputWidth: Int = 32       // Output width
)
```

### Configuration Example

```scala
val bondConfig = BondParam(
  bondType = "vvv",
  inputWidth = 8,
  outputWidth = 32
)

val threadConfig = ThreadParam(
  lane = 16,
  attr = "vector",
  threadName = "mul_thread",
  Op = OpParam("mul", bondConfig)
)
```

## Usage

### Creating VVV Binding

```scala
// Using VVV binding in thread
class CustomThread(implicit p: Parameters) extends BaseThread
  with CanHaveVVVBond {

  // Get binding interface
  for (bond <- vvvBond) {
    // Connect input
    bond.in.valid := inputValid
    bond.in.bits.in1 := inputVector1
    bond.in.bits.in2 := inputVector2

    // Connect output
    outputValid := bond.out.valid
    outputVector := bond.out.bits.out
    bond.out.ready := outputReady
  }
}
```

### Binding Connection

```scala
// Connect binding interfaces of two modules
val producer = Module(new ProducerThread())
val consumer = Module(new ConsumerThread())

// Direct binding interface connection
for {
  prodBond <- producer.vvvBond
  consBond <- consumer.vvvBond
} {
  consBond.in <> prodBond.out
}
```

## Synchronization Mechanisms

### Handshake Protocol

VVV binding uses standard Decoupled handshake protocol:

1. **Data preparation**: Producer sets `valid` and `bits`
2. **Receive ready**: Consumer sets `ready`
3. **Data transmission**: Transfer completes when `valid && ready`
4. **State update**: Both sides update internal state

### Backpressure Handling

Binding interface supports backpressure mechanism:

```scala
// When downstream is not ready, upstream waits
when(!downstream.ready) {
  upstream.valid := false.B
  // Keep data unchanged
}
```

## Extensibility

### New Binding Types

New binding types can be defined following a similar pattern:

```scala
// Single input single output binding
class VV(implicit p: Parameters) extends Bundle {
  val in = Flipped(Decoupled(Vec(lane, UInt(inputWidth.W))))
  val out = Decoupled(Vec(lane, UInt(outputWidth.W)))
}

// Corresponding trait
trait CanHaveVVBond { this: BaseThread =>
  val vvBond = params(ThreadBondKey).filter(_.bondType == "vv").map { _ =>
    IO(new VV()(params))
  }
}
```

### Parameterization Support

The binding module supports full parameterized configuration:

- Vector lane count configurable
- Input/output width configurable
- Binding type extensible

## Related Modules

- [Thread Module](../thread/README.md) - Provides usage environment for bindings
- [Vector Operations Module](../op/README.md) - Data processing logic for bindings
- [Vector Processing Unit](../README.md) - Upper-level vector processor
