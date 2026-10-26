# Vector Operations Module

## Overview

The vector operations module implements specific computation operations in the vector processing unit, located at `prototype/vector/op`. This module provides implementations of different types of vector operations, including multiplication operations and cascade operations.

## File Structure

```
op/
├── cascade.scala    - Cascade addition operation
└── mul.scala       - Multiplication operation
```

## Core Components

### CascadeOp - Cascade Addition Operation

CascadeOp implements element-wise addition operation on vector elements:

```scala
class CascadeOp(implicit p: Parameters) extends Module {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val outputWidth = bondParam.outputWidth

  val io = IO(new VVV()(p))
}
```

#### Operation Logic

```scala
val reg1 = RegInit(VecInit(Seq.fill(lane)(0.U(outputWidth.W))))
val valid1 = RegInit(false.B)

when (io.in.valid) {
  valid1 := true.B
  reg1 := io.in.bits.in1.zip(io.in.bits.in2).map { case (a, b) => a + b }
}
```

**Function description**:
- Receives two input vectors `in1` and `in2`
- Performs element-wise addition: `out[i] = in1[i] + in2[i]`
- Uses register to cache computation results
- Supports pipelined operations

#### Flow Control Mechanism

```scala
io.in.ready := io.out.ready

when (io.out.ready && valid) {
  io.out.valid := true.B
  io.out.bits.out := reg1
}.otherwise {
  io.out.valid := false.B
  io.out.bits.out := VecInit(Seq.fill(lane)(0.U(outputWidth.W)))
}
```

### MulOp - Multiplication Operation

MulOp implements vector multiplication operation with broadcast mode support:

```scala
class MulOp(implicit p: Parameters) extends Module {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val inputWidth = bondParam.inputWidth

  val io = IO(new VVV()(p))
}
```

#### Operation Logic

```scala
val reg1 = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
val reg2 = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
val cnt = RegInit(0.U(log2Ceil(lane).W))
val active = RegInit(false.B)

when (io.in.valid) {
  reg1 := io.in.bits.in1
  reg2 := io.in.bits.in2
  cnt := 0.U
  active := true.B
}
```

**Function description**:
- Receives two input vectors and caches them in registers
- Uses counter `cnt` to control output sequence
- Implements broadcast multiplication: `out[i] = reg1[cnt] * reg2[i]`

#### Sequential Output

```scala
for (i <- 0 until lane) {
  io.out.bits.out(i) := reg1(cnt) * reg2(i)
}

when (active && io.out.ready) {
  cnt := cnt + 1.U
  when (cnt === (lane-1).U) {
    active := false.B
  }
}
```

**Output mode**:
- Outputs one set of multiplication results per cycle
- `reg1[cnt]` multiplied with all elements of `reg2`
- Counter increments to achieve sequential output

## Operation Traits

### CanHaveCascadeOp - Cascade Operation Trait

```scala
trait CanHaveCascadeOp { this: BaseThread =>
  val cascadeOp = params(ThreadOpKey).filter(_.OpType == "cascade").map { opParam =>
    Module(new CascadeOp()(params))
  }

  def getCascadeOp = cascadeOp
}
```

### CanHaveMulOp - Multiplication Operation Trait

```scala
trait CanHaveMulOp { this: BaseThread =>
  val mulOp = params(ThreadOpKey).filter(_.OpType == "mul").map { opParam =>
    Module(new MulOp()(params))
  }

  def getMulOp = mulOp
}
```

## Usage

### Using Operations in Threads

```scala
class CasThread(implicit p: Parameters) extends BaseThread
  with CanHaveCascadeOp
  with CanHaveVVVBond {

  // Connect operation and binding
  for {
    op <- cascadeOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

### Configuring Operation Parameters

```scala
val opParam = OpParam(
  OpType = "cascade",                    // Operation type
  bondType = BondParam(
    bondType = "vvv",
    inputWidth = 32,
    outputWidth = 32
  )
)
```

## Operation Type Comparison

### CascadeOp vs MulOp

| Feature | CascadeOp | MulOp |
|------|-----------|-------|
| Operation type | Element-wise addition | Broadcast multiplication |
| Input width | Arbitrary | Usually smaller |
| Output width | Arbitrary | Usually larger |
| Latency | 1 cycle | lane cycles |
| Throughput | 1 group per cycle | 1 group per lane cycle |
| Resource consumption | Adder × lane | Multiplier × lane |

### Application Scenarios

**CascadeOp is suitable for**:
- Vector addition operations
- Accumulation operations
- Data merging

**MulOp is suitable for**:
- Matrix-vector multiplication
- Convolution operations
- Scaling operations

## Data Flow Patterns

### CascadeOp Data Flow

```
Input: [a0, a1, ..., an], [b0, b1, ..., bn]
      ↓
Compute: [a0+b0, a1+b1, ..., an+bn]
      ↓
Output: [c0, c1, ..., cn] (1 cycle)
```

### MulOp Data Flow

```
Input: [a0, a1, ..., an], [b0, b1, ..., bn]
      ↓
Cycle 0: [a0*b0, a0*b1, ..., a0*bn]
Cycle 1: [a1*b0, a1*b1, ..., a1*bn]
...
Cycle n: [an*b0, an*b1, ..., an*bn]
```

## Extended Operations

### Adding New Operations

New vector operations can be added following a similar pattern:

```scala
class SubOp(implicit p: Parameters) extends Module {
  val io = IO(new VVV()(p))

  // Implement subtraction operation
  io.out.bits.out := io.in.bits.in1.zip(io.in.bits.in2).map {
    case (a, b) => a - b
  }
}

trait CanHaveSubOp { this: BaseThread =>
  val subOp = params(ThreadOpKey).filter(_.OpType == "sub").map { _ =>
    Module(new SubOp()(params))
  }
}
```

### Complex Operations

For more complex operations, multiple basic operations can be combined:

```scala
class FMAOp(implicit p: Parameters) extends Module {
  // Fused multiply-add operation: out = a * b + c
  val mulOp = Module(new MulOp())
  val addOp = Module(new CascadeOp())

  // Connect operation pipeline
  addOp.io.in.bits.in1 <> mulOp.io.out.bits.out
  // ...
}
```

## Performance Optimization

### Pipeline Optimization

- Use registers to cache intermediate results
- Support continuous data stream processing
- Minimize combinational logic delay

### Resource Optimization

- Choose appropriate hardware resources based on operation type
- Support resource sharing and reuse
- Configurable parallelism

## Related Modules

- [Binding Module](../bond/README.md) - Provides data interfaces
- [Thread Module](../thread/README.md) - Provides execution environment for operations
- [Vector Processing Unit](../README.md) - Upper-level vector processor
