# Matrix Computation Accelerator

## Overview

This directory implements Buckyball's matrix computation accelerator for matrix multiplication and related operations. Located at `arch/src/main/scala/prototype/matrix`, it serves as a matrix computation accelerator supporting multiple data formats and operation modes.

Core components:
- **bbfp_control.scala**: Matrix computation controller
- **bbfp_pe.scala**: Processing Element (PE) and MAC unit
- **bbfp_buffer.scala**: Data buffer management
- **bbfp_load.scala**: Data load unit
- **bbfp_ex.scala**: Execution unit
- **bbfpIns_decode.scala**: Instruction decoder

## Code Structure

```
matrix/
├── bbfp_control.scala   - Controller main module
├── bbfp_pe.scala        - Processing element implementation
├── bbfp_buffer.scala    - Buffer management
├── bbfp_load.scala      - Load unit
├── bbfp_ex.scala        - Execution unit
└── bbfpIns_decode.scala - Instruction decode
```

### File Dependencies

**bbfp_control.scala** (Controller layer)
- Integrates submodules (ID, LU, EX, etc.)
- Manages SRAM and Accumulator interfaces
- Handles Ball domain commands

**bbfp_pe.scala** (Computation core layer)
- Implements MacUnit multiply-accumulate unit
- Defines PEControl control signals
- Handles signed/unsigned operations

**Other modules** (Functional support layer)
- Provides data buffering, loading, execution and other support functions

## Module Description

### bbfp_control.scala

**Main functionality**: Top-level control module for matrix computation accelerator

**Module integration**:
```scala
class BBFP_Control extends Module {
  val BBFP_ID = Module(new BBFP_ID)
  val ID_LU = Module(new ID_LU)
  val BBFP_LoadUnit = Module(new BBFP_LoadUnit)
  val LU_EX = Module(new LU_EX)
}
```

**Interface definition**:
```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)
  val is_matmul_ws = Input(Bool())
  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
  val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(...)))
  val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(...)))
})
```

**Data flow**:
```
cmdReq → BBFP_ID → ID_LU → BBFP_LoadUnit → LU_EX
                              ↓
                         SRAM/ACC interface
```

### bbfp_pe.scala

**Main functionality**: Implements basic processing element for matrix computation

**MAC unit definition**:
```scala
class MacUnit extends Module {
  val io = IO(new Bundle {
    val in_a = Input(UInt(7.W))    // [6]=sign, [5]=flag, [4:0]=value
    val in_b = Input(UInt(7.W))    // [6]=sign, [5]=flag, [4:0]=value
    val in_c = Input(UInt(32.W))   // [31]=sign, [30:0]=value
    val out_d = Output(UInt(32.W)) // Output result
  })
}
```

**Data format processing**:
```scala
// Extract sign bit and value
val sign_a = io.in_a(6)
val sign_b = io.in_b(6)
val flag_a = io.in_a(5)
val flag_b = io.in_b(5)
val value_a = io.in_a(4, 0)
val value_b = io.in_b(4, 0)

// Determine left shift based on flag bit
val shifted_a = Mux(flag_a === 1.U, value_a << 2, value_a)
val shifted_b = Mux(flag_b === 1.U, value_b << 2, value_b)
```

**Signed arithmetic**:
```scala
val a_signed = Mux(sign_a === 1.U, -(shifted_a.zext), shifted_a.zext).asSInt
val b_signed = Mux(sign_b === 1.U, -(shifted_b.zext), shifted_b.zext).asSInt
```

**Control signals**:
```scala
class PEControl extends Bundle {
  val propagate = UInt(1.W)   // Propagation control
}
```

## Usage

### Data Format

**Input format**: 7-bit compressed format
- bit[6]: Sign bit (0=positive, 1=negative)
- bit[5]: Flag bit (1=left shift by 2)
- bit[4:0]: 5-bit value

**Output format**: 32-bit signed number
- bit[31]: Sign bit
- bit[30:0]: 31-bit value

### Operation Characteristics

**MAC operation**: Multiply-Accumulate operation
- Supports signed and unsigned operations
- Configurable shift operations
- 32-bit accumulator output

**Pipeline structure**:
- ID: Instruction decode stage
- LU: Load unit stage
- EX: Execution unit stage

### Notes

1. **Data format**: Uses custom 7-bit compressed format to reduce storage overhead
2. **Sign handling**: Supports correct signed number operations and sign extension
3. **Shift optimization**: Controls data preprocessing shift through flag bit
4. **Interface compatibility**: Fully compatible with SRAM and Accumulator interfaces
5. **Pipeline design**: Multi-stage pipeline improves throughput
