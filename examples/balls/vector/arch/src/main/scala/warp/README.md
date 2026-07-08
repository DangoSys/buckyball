# Thread Warp Module

## Overview

The thread warp module implements thread warp management functionality in the vector processing unit, located at `prototype/vector/warp`. This module organizes multiple threads into a mesh structure, implementing parallel computation and dataflow management.

## File Structure

```
warp/
├── MeshWarp.scala    - Mesh warp implementation
└── VecBall.scala     - Vector ball processor
```

## Core Components

### MeshWarp - Mesh Warp

MeshWarp implements a 32-thread mesh structure containing 16 multiplication threads and 16 cascade threads:

```scala
class MeshWarp(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new MeshWarpInput))
    val out = Decoupled(new MeshWarpOutput)
  })
}
```

#### Input/Output Interface

```scala
class MeshWarpInput extends Bundle {
  val op1 = Vec(16, UInt(8.W))        // First operand vector
  val op2 = Vec(16, UInt(8.W))        // Second operand vector
  val thread_id = UInt(10.W)          // Thread identifier
}

class MeshWarpOutput extends Bundle {
  val res = Vec(16, UInt(32.W))       // Result vector
}
```

#### Thread Configuration

Threads in the mesh are configured according to the following rules:

```scala
val threadMap = (0 until 32).map { i =>
  val threadName = i.toString
  val opType = if (i < 16) "mul" else "cascade"
  val bond = if (opType == "mul") {
    BondParam("vvv", inputWidth = 8, outputWidth = 32)
  } else {
    BondParam("vvv", inputWidth = 32, outputWidth = 32)
  }
  val op = OpParam(opType, bond)
  val thread = ThreadParam(16, s"attr$threadName", threadName, op)
  threadName -> thread
}.toMap
```

Thread allocation:
- Threads 0-15: Multiplication operation threads (8-bit input → 32-bit output)
- Threads 16-31: Cascade operation threads (32-bit input → 32-bit output)

#### Data Flow Connection

Data flow in the mesh is connected as follows:

```scala
// Connect mul thread output to cascade thread input
casBond.in.bits.in1 := mulBond.out.bits.out
mulBond.out.ready   := casBond.in.ready

// Cascade connection between cascade threads
if (i == 0) {
  casBond.in.bits.in2 := VecInit(Seq.fill(16)(0.U(32.W)))
} else {
  casBond.in.bits.in2 := prevCasBond.out.bits.out
}
```

Data flow path:
1. Input data → Multiplication threads (thread 0-15)
2. Multiplication results → Cascade threads (thread 16-31)
3. Serial connection between cascade threads
4. Final result output from thread 31

### VecBall - Vector Ball Processor

VecBall is a wrapper for MeshWarp, providing state management and iteration control:

```scala
class VecBall(implicit p: Parameters) extends Module {
  val io = IO(new VecBallIO())
}
```

#### Interface Definition

```scala
class VecBallIO extends BallIO {
  val op1In = Flipped(Valid(Vec(16, UInt(8.W))))    // Operand 1 input
  val op2In = Flipped(Valid(Vec(16, UInt(8.W))))    // Operand 2 input
  val rstOut = Decoupled(Vec(16, UInt(32.W)))       // Result output
}

class BallIO extends Bundle {
  val iterIn = Flipped(Decoupled(UInt(10.W)))       // Iteration count input
  val iterOut = Valid(UInt(10.W))                   // Current iteration output
}
```

#### State Management

VecBall maintains the following internal state:

```scala
val start  = RegInit(false.B)      // Start flag
val arrive = RegInit(false.B)      // Arrival flag
val done   = RegInit(false.B)      // Completion flag
val iter   = RegInit(0.U(10.W))    // Total iteration count
val iterCounter = RegInit(0.U(10.W)) // Current iteration counter
```

#### Thread Scheduling

VecBall uses round-robin scheduling to assign threads:

```scala
val threadId = RegInit(0.U(4.W))
when (io.op1In.valid && io.op2In.valid && threadId < 15.U) {
  threadId := threadId + 1.U
} .elsewhen (io.op1In.valid && io.op2In.valid && threadId === 15.U) {
  threadId := 0.U
}
```

## Usage

### Creating MeshWarp Instance

```scala
val meshWarp = Module(new MeshWarp()(p))

// Connect input
meshWarp.io.in.valid := inputValid
meshWarp.io.in.bits.op1 := operand1
meshWarp.io.in.bits.op2 := operand2
meshWarp.io.in.bits.thread_id := selectedThread

// Connect output
outputValid := meshWarp.io.out.valid
result := meshWarp.io.out.bits.res
meshWarp.io.out.ready := outputReady
```

### Creating VecBall Instance

```scala
val vecBall = Module(new VecBall()(p))

// Set iteration count
vecBall.io.iterIn.valid := iterValid
vecBall.io.iterIn.bits := totalIterations

// Input data
vecBall.io.op1In.valid := dataValid
vecBall.io.op1In.bits := inputVector1
vecBall.io.op2In.valid := dataValid
vecBall.io.op2In.bits := inputVector2

// Get result
outputReady := vecBall.io.rstOut.ready
when(vecBall.io.rstOut.valid) {
  result := vecBall.io.rstOut.bits
}
```

## Computation Modes

### Vector Multiply-Accumulate

Computation mode implemented by MeshWarp:

1. **Multiplication phase**: 16 multiplication threads compute `op1[i] * op2[i]` in parallel
2. **Accumulation phase**: 16 cascade threads accumulate multiplication results serially
3. **Output phase**: Output final accumulated vector

### Iterative Processing

VecBall supports multi-iteration processing:

1. Set iteration count `iterIn`
2. Loop input data pairs
3. Monitor iteration count `iterOut`
4. Check completion status

## Performance Characteristics

- **Parallelism**: 16 multiplication operations execute in parallel
- **Pipeline**: Supports continuous data stream processing
- **Throughput**: Can process one 16-element vector pair per cycle
- **Latency**: Combined latency of multiplication + cascade

## Related Modules

- [Thread Module](../thread/README.md) - Provides basic thread implementation
- [Vector Operations Module](../op/README.md) - Provides multiplication and cascade operations
- [Binding Module](../bond/README.md) - Provides data interfaces
