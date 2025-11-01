# BallDomain Example Implementation

## Overview

This directory contains a complete example implementation of BallDomain in the BuckyBall framework, demonstrating how to build a custom computation domain to manage specialized accelerators. BallDomain is a core concept in BuckyBall architecture, used to encapsulate and manage a group of related computation units with unified control and dataflow management.

This directory implements the ball domain architecture, including:
- **BallDomain**: Top-level module managing the entire computation domain
- **BallController**: Ball domain controller for instruction scheduling and execution control
- **DISA**: Distributed instruction scheduling architecture
- **DomainDecoder**: Domain instruction decoder
- **Specialized accelerators**: Including matrix, vector, im2col and other accelerator implementations

## Code Structure

```
balldomain/
├── BallDomain.scala      - Ball domain top module
├── BallController.scala  - Ball domain controller
├── DISA.scala           - Distributed instruction scheduling architecture
├── DomainDecoder.scala  - Domain instruction decoder
├── bbus/                - Ball domain bus system
├── im2col/              - Image-to-column conversion accelerator
├── matrixball/          - Matrix computation ball domain
├── rs/                  - Reservation station implementation
└── vecball/             - Vector computation ball domain
```

### File Dependencies

**BallDomain.scala** (Top-level module)
- Integrates all submodules, provides unified ball domain interface
- Manages dataflow and control flow within ball domain
- Connects to system bus and RoCC interface

**BallController.scala** (Control layer)
- Implements instruction scheduling and execution control for ball domain
- Manages coordination between multiple accelerators
- Provides state management and error handling

**DISA.scala** (Scheduling layer)
- Distributed instruction scheduling architecture implementation
- Supports concurrent execution of multiple instructions
- Provides dynamic load balancing

**DomainDecoder.scala** (Decode layer)
- Ball domain specific instruction decode
- Instruction dispatch to corresponding execution units
- Supports complex instruction decomposition and reorganization

## Module Description

### BallDomain.scala

**Main functionality**: Ball domain top module, integrates all computation units and control logic

**Key components**:

```scala
class BallDomain(implicit p: Parameters) extends LazyModule {
  val controller = LazyModule(new BallController)
  val matrixBall = LazyModule(new MatrixBall)
  val vecBall = LazyModule(new VecBall)
  val im2colUnit = LazyModule(new Im2colUnit)

  // Ball domain bus connections
  val bbus = LazyModule(new BBus)
  bbus.node := controller.node
  matrixBall.node := bbus.node
  vecBall.node := bbus.node
}
```

**Inputs/Outputs**:
- Input: RoCC instruction interface, memory access interface
- Output: Computation results, status information
- Edge cases: Instruction conflict handling, resource contention management

### BallController.scala

**Main functionality**: Ball domain controller, manages overall ball domain execution control

**Key components**:

```scala
class BallController extends Module {
  val io = IO(new Bundle {
    val rocc = Flipped(new RoCCCoreIO)
    val mem = new HellaCacheIO
    val domain_ctrl = new DomainControlIO
  })

  // Instruction queue and scheduling logic
  val inst_queue = Module(new Queue(new RoCCInstruction, 16))
  val scheduler = Module(new InstructionScheduler)
}
```

**Scheduling strategy**:
- Static scheduling based on instruction type
- Dynamic resource allocation and load balancing
- Supports instruction pipelining and concurrent execution

### DISA.scala

**Main functionality**: Distributed instruction scheduling architecture

**Key components**:

```scala
class DISA extends Module {
  val io = IO(new Bundle {
    val inst_in = Flipped(Decoupled(new Instruction))
    val exec_units = Vec(numUnits, new ExecutionUnitIO)
    val completion = Decoupled(new CompletionInfo)
  })

  // Distributed dispatch table
  val dispatch_table = Reg(Vec(numUnits, new DispatchEntry))
  val load_balancer = Module(new LoadBalancer)
}
```

**Scheduling algorithms**:
- Round-robin scheduling for fairness
- Priority scheduling for critical tasks
- Dynamic scheduling adapts to load changes

### DomainDecoder.scala

**Main functionality**: Ball domain instruction decoder

**Key components**:

```scala
class DomainDecoder extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(32.W))
    val decoded = Output(new DecodedInstruction)
    val valid = Output(Bool())
  })

  // Instruction decode table
  val decode_table = Array(
    MATRIX_OP -> MatrixOpDecoder,
    VECTOR_OP -> VectorOpDecoder,
    IM2COL_OP -> Im2colOpDecoder
  )
}
```

**Decode functionality**:
- Supports multiple instruction formats
- Microcode expansion for complex instructions
- Instruction dependency analysis and optimization

## Usage

### Design Features

1. **Modular architecture**: Each accelerator is an independent module, easy to extend and maintain
2. **Unified interface**: All accelerators communicate through unified ball domain bus
3. **Flexible scheduling**: Supports multiple scheduling strategies, adapts to different computation patterns
4. **Scalability**: Easy to add new accelerator types and functionality

### Performance Optimization

1. **Pipeline design**: Instruction decode, scheduling, execution use pipeline architecture
2. **Concurrent execution**: Supports multiple accelerators working simultaneously
3. **Data management**: Data caching and access management
4. **Workload**: Workload distribution

### Usage Example

```scala
// Create ball domain instance
val ballDomain = LazyModule(new BallDomain)

// Connect to RoCC interface
rocc.cmd <> ballDomain.module.io.rocc.cmd
rocc.resp <> ballDomain.module.io.rocc.resp

// Configure ball domain parameters
ballDomain.module.io.config := ballDomainConfig
```

### Notes

1. **Resource management**: Properly allocate computational resources, avoid resource conflicts
2. **Timing constraints**: Pay attention to timing relationships and data synchronization between different modules
3. **Power control**: Implement dynamic power management, shut down modules when not in use
4. **Debug support**: Debug interface and status monitoring functionality
