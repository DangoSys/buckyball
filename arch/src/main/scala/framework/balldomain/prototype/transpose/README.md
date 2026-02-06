# Matrix Transpose Accelerator

## Overview

This directory implements Buckyball's matrix transpose accelerator for matrix transpose operations. Located at `arch/src/main/scala/prototype/transpose`, it serves as a matrix transpose accelerator supporting pipelined transpose operations.

Core components:
- **Transpose.scala**: Pipelined transposer implementation

## Code Structure

```
transpose/
└── Transpose.scala  - Pipelined transposer
```

### Module Responsibilities

**Transpose.scala** (Transpose implementation layer)
- Implements PipelinedTransposer module
- Manages matrix data read, transpose, and write-back
- Provides Ball domain command interface

## Module Description

### Transpose.scala

**Main functionality**: Implements pipelined matrix transpose operation

**State machine definition**:
```scala
val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
val state = RegInit(idle)
```

**Storage structure**:
```scala
// Matrix storage register (veclane x veclane)
val regArray = Reg(Vec(b.veclane, Vec(b.veclane, UInt(b.inputType.getWidth.W))))
```

**Counter management**:
```scala
val readCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
val respCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
```

**Instruction registers**:
```scala
val robid_reg = RegInit(0.U(10.W))    // ROB ID
val waddr_reg = RegInit(0.U(10.W))    // Write address
val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))  // Write bank
val raddr_reg = RegInit(0.U(10.W))    // Read address
val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))  // Read bank
val iter_reg = RegInit(0.U(10.W))     // Iteration count
```

**Interface definition**:
```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)
  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
})
```

**Processing flow**:
1. **idle**: Wait for command, parse transpose parameters
2. **sRead**: Read matrix data row by row into register array
3. **sWrite**: Write transposed data column by column
4. **complete**: Send completion signal

**Transpose algorithm**:
- Uses veclane×veclane register array to store matrix
- Reads row-wise, writes column-wise to implement transpose
- Supports block-wise transpose for matrices of arbitrary size

## Usage

### Implementation Details

**State machine**:
```scala
val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
```
- `idle`: Wait for instruction
- `sRead`: Read matrix data
- `sWrite`: Write transpose result
- `complete`: Complete and respond

**Register array**:
```scala
val regArray = Reg(Vec(b.veclane, Vec(b.veclane, UInt(b.inputType.getWidth.W))))
```
Uses veclane×veclane register array to cache matrix data.

**Transpose operation**:
- Read phase: Read data row by row into `regArray(row)(col)`
- Write phase: Read `regArray(i)(col)` column by column to form new rows for writing

### Configuration Parameters

**Matrix size**: Determined by b.veclane parameter
**Data width**: Determined by b.inputType.getWidth
**Bank configuration**: Supports multi-bank SRAM access

### Notes

1. **Matrix size limitation**: Maximum support for veclane×veclane matrices
2. **Memory bandwidth**: Transpose operation has high memory bandwidth requirements
3. **Register overhead**: Requires veclane² registers to store matrix
4. **Address calculation**: Transposed address calculation needs to be handled correctly
5. **Pipeline control**: Read/write counters need to be synchronized correctly
