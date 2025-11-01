# Vector Processing Unit

## Overview

The Vector Processing Unit is a specialized computation accelerator in the BuckyBall framework, located at `prototype/vector`. This module implements a complete vector processing pipeline, including control unit, load unit, execution unit, and store unit, supporting parallel processing of vector data.

## File Structure

```
vector/
├── VecUnit.scala         - Vector processing unit top module
├── VecCtrlUnit.scala     - Vector control unit
├── VecLoadUnit.scala     - Vector load unit
├── VecEXUnit.scala       - Vector execution unit
├── VecStoreUnit.scala    - Vector store unit
├── bond/                 - Binding and synchronization mechanisms
├── op/                   - Vector operation implementations
├── thread/               - Thread management
└── warp/                 - Thread warp management
```

## Core Components

### VecUnit - Vector Processing Unit Top Level

VecUnit is the top-level module of the vector processor, integrating all sub-units:

```scala
class VecUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // Connected to Scratchpad SRAM read/write interfaces
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
    // Connected to Accumulator read/write interfaces
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })
}
```

#### Interface Description

**Command interface**:
- `cmdReq`: Vector instruction request from reservation station
- `cmdResp`: Completion response returned to reservation station

**Memory interface**:
- `sramRead/sramWrite`: Read/write interfaces connected to Scratchpad
- `accRead/accWrite`: Read/write interfaces connected to Accumulator

### VecCtrlUnit - Vector Control Unit

The vector control unit is responsible for instruction decode and pipeline control:

```scala
class VecCtrlUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle{
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp_o = Decoupled(new BallRsComplete)

    val ctrl_ld_o = Decoupled(new ctrl_ld_req)
    val ctrl_st_o = Decoupled(new ctrl_st_req)
    val ctrl_ex_o = Decoupled(new ctrl_ex_req)

    val cmdResp_i = Flipped(Valid(new Bundle {val commit = Bool()}))
  })
}
```

#### Control State

```scala
val rob_id_reg    = RegInit(0.U(log2Up(b.rob_entries).W))
val iter          = RegInit(0.U(10.W))
val op1_bank      = RegInit(0.U(2.W))
val op1_bank_addr = RegInit(0.U(12.W))
val op2_bank_addr = RegInit(0.U(12.W))
val op2_bank      = RegInit(0.U(2.W))
val wr_bank       = RegInit(0.U(2.W))
val wr_bank_addr  = RegInit(0.U(12.W))
val is_acc        = RegInit(false.B)
```

### Data Flow Architecture

The vector processing unit uses a pipeline architecture with the following data flow:

```
Instruction input → VecCtrlUnit → Control signal dispatch
                          ↓
                  VecLoadUnit (Load data)
                          ↓
                  VecEXUnit (Execute computation)
                          ↓
                  VecStoreUnit (Store results)
                          ↓
                      Completion response
```

#### Module Connections

```scala
// Control unit
val VecCtrlUnit = Module(new VecCtrlUnit)
VecCtrlUnit.io.cmdReq <> io.cmdReq
io.cmdResp <> VecCtrlUnit.io.cmdResp_o

// Load unit
val VecLoadUnit = Module(new VecLoadUnit)
VecLoadUnit.io.ctrl_ld_i <> VecCtrlUnit.io.ctrl_ld_o

// Execution unit
val VecEX = Module(new VecEXUnit)
VecEX.io.ctrl_ex_i <> VecCtrlUnit.io.ctrl_ex_o
VecEX.io.ld_ex_i <> VecLoadUnit.io.ld_ex_o

// Store unit
val VecStoreUnit = Module(new VecStoreUnit)
VecStoreUnit.io.ctrl_st_i <> VecCtrlUnit.io.ctrl_st_o
VecStoreUnit.io.ex_st_i <> VecEX.io.ex_st_o
```

## Memory System Integration

### Scratchpad Connection

The vector processing unit connects to Scratchpad through multiple banks:

```scala
for (i <- 0 until b.sp_banks) {
  io.sramRead(i).req <> VecLoadUnit.io.sramReadReq(i)
  VecLoadUnit.io.sramReadResp(i) <> io.sramRead(i).resp
}
```

### Accumulator Connection

Execution results are written to Accumulator through the store unit:

```scala
for (i <- 0 until b.acc_banks) {
  io.accWrite(i) <> VecStoreUnit.io.accWrite(i)
}
```

## Configuration Parameters

### Vector Configuration

Configure vector processor parameters through `CustomBuckyBallConfig`:

```scala
class CustomBuckyBallConfig extends Config((site, here, up) => {
  case "veclane" => 16              // Vector lane count
  case "sp_banks" => 4              // Scratchpad bank count
  case "acc_banks" => 2             // Accumulator bank count
  case "spad_bank_entries" => 1024  // Entries per bank
  case "acc_bank_entries" => 512    // Accumulator entry count
})
```

### Data Width

```scala
val spad_w = b.veclane * b.inputType.getWidth  // Scratchpad width
val acc_w = b.outputType.getWidth              // Accumulator width
```

## Usage

### Creating Vector Processing Unit

```scala
val vecUnit = Module(new VecUnit())

// Connect command interface
vecUnit.io.cmdReq <> reservationStation.io.issue
reservationStation.io.complete <> vecUnit.io.cmdResp

// Connect memory system
for (i <- 0 until sp_banks) {
  scratchpad.io.read(i) <> vecUnit.io.sramRead(i)
  scratchpad.io.write(i) <> vecUnit.io.sramWrite(i)
}

for (i <- 0 until acc_banks) {
  accumulator.io.read(i) <> vecUnit.io.accRead(i)
  accumulator.io.write(i) <> vecUnit.io.accWrite(i)
}
```

### Vector Instruction Format

Vector instructions are passed through the `BallRsIssue` interface:

```scala
class BallRsIssue extends Bundle {
  val cmd = new Bundle {
    val iter = UInt(10.W)           // Iteration count
    val op1_bank = UInt(2.W)        // Operand 1 bank
    val op1_bank_addr = UInt(12.W)  // Operand 1 address
    val op2_bank = UInt(2.W)        // Operand 2 bank
    val op2_bank_addr = UInt(12.W)  // Operand 2 address
    val wr_bank = UInt(2.W)         // Write bank
    val wr_bank_addr = UInt(12.W)   // Write address
  }
  val rob_id = UInt(log2Up(rob_entries).W)
}
```

## Execution Model

### Pipeline Execution

1. **Instruction decode**: VecCtrlUnit decodes vector instructions
2. **Data load**: VecLoadUnit loads operands from Scratchpad
3. **Vector computation**: VecEXUnit executes vector operations
4. **Result store**: VecStoreUnit writes results to Accumulator
5. **Completion response**: Returns completion signal to reservation station

### Parallel Processing

- **Multi-lane parallelism**: Supports parallel computation across multiple vector lanes
- **Bank-level parallelism**: Multiple memory banks support parallel access
- **Pipeline overlap**: Different stages can overlap execution

## Submodule Description

### Binding Mechanism (Bond)
Provides inter-thread synchronization and data binding functionality, supporting producer-consumer pattern data transfer.

### Vector Operations (Op)
Implements specific vector computation operations, including arithmetic operations, logical operations, and special functions.

### Thread Management (Thread)
Provides thread abstraction and management functionality, supporting different types of vector threads.

### Thread Warp Management (Warp)
Implements thread warp organization and scheduling, supporting large-scale parallel computation.

## Performance Characteristics

- **High parallelism**: Supports multi-lane vector parallel processing
- **Pipelined**: Multi-stage pipeline improves throughput
- **Memory optimization**: Multi-bank memory system reduces access conflicts
- **Flexible configuration**: Supports different vector lengths and data types

## Related Modules

- [Binding Mechanism](bond/README.md) - Thread synchronization and data binding
- [Vector Operations](op/README.md) - Specific computation operation implementations
- [Thread Management](thread/README.md) - Thread abstraction and management
- [Thread Warp Management](warp/README.md) - Thread warp organization and scheduling
- [Prototype Accelerator Overview](../README.md) - Upper-level accelerator framework
