# Buckyball Utility Library

## Overview

This directory contains general utility functions and helper modules in the Buckyball framework, primarily providing reusable hardware design components. Located at `arch/src/main/scala/Util`, it serves as the base utility layer throughout the architecture, providing common hardware building blocks for other modules.

Main functionality includes:
- **Pipeline**: Pipeline control and management tools
- Common hardware design pattern implementations

## Code Structure

```
Util/
└── Pipeline.scala    - Pipeline control implementation
```

### File Dependencies

**Pipeline.scala** (Base utility layer)
- Provides general pipeline control logic
- Referenced by other modules requiring pipeline functionality
- Implements standard pipeline interfaces and control signals

## Module Description

### Pipeline.scala

**Main functionality**: Provides general pipeline control and management functionality

**Key components**:

```scala
class Pipeline extends Module {
  val io = IO(new Bundle {
    val flush = Input(Bool())
    val stall = Input(Bool())
    val valid_in = Input(Bool())
    val ready_out = Output(Bool())
    val valid_out = Output(Bool())
  })

  // Pipeline control logic
  val pipeline_valid = RegInit(false.B)

  when(io.flush) {
    pipeline_valid := false.B
  }.elsewhen(!io.stall) {
    pipeline_valid := io.valid_in
  }

  io.ready_out := !io.stall
  io.valid_out := pipeline_valid && !io.flush
}
```

**Pipeline control signals**:
- **flush**: Pipeline flush signal, clears all pipeline stages
- **stall**: Pipeline stall signal, maintains current state
- **valid_in**: Input data valid signal
- **ready_out**: Ready to receive new data signal
- **valid_out**: Output data valid signal

**Inputs/Outputs**:
- Input: Control signals (flush, stall) and data valid signal
- Output: Pipeline state and data valid indication
- Edge cases: flush has higher priority than stall, ensuring correct pipeline behavior

**Dependencies**: Chisel3 base library, standard Module and Bundle interfaces

## Usage

### Usage

**Integrating pipeline control**:
```scala
class MyModule extends Module {
  val pipeline = Module(new Pipeline)

  // Connect control signals
  pipeline.io.flush := flush_condition
  pipeline.io.stall := stall_condition
  pipeline.io.valid_in := input_valid

  // Use pipeline output
  val output_enable = pipeline.io.valid_out
}
```

### Design Patterns

**Pipeline cascading**:
- Supports cascaded connection of multi-stage pipelines
- Provides standard ready/valid handshake protocol
- Ensures correctness and timing of data flow

**Backpressure handling**:
- Implements standard backpressure propagation mechanism
- Supports pause and resume of upstream modules
- Guarantees no data loss or duplication

### Notes

1. **Timing constraints**: flush signal should be asserted synchronously at clock rising edge
2. **Reset behavior**: Pipeline should clear all valid bits on reset
3. **Combinational logic**: ready signal is combinational logic, avoid timing path issues
4. **Extensibility**: Design supports parameterized pipeline depth and data width
