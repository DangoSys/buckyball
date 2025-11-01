# Im2col Image Processing Accelerator

## Overview

This directory implements BuckyBall's Im2col operation accelerator for image-to-column matrix conversion in convolutional neural networks. Located at `arch/src/main/scala/prototype/im2col`, it serves as an image processing accelerator that converts convolution operations to matrix multiplication operations to improve computational efficiency.

Core components:
- **im2col.scala**: Im2col accelerator main implementation

## Code Structure

```
im2col/
└── im2col.scala  - Im2col accelerator implementation
```

### Module Responsibilities

**Im2col.scala** (Accelerator implementation layer)
- Implements image-to-column matrix conversion logic
- Manages SRAM read/write operations
- Provides Ball domain command interface

## Module Description

### im2col.scala

**Main functionality**: Implements sliding convolution window and data rearrangement

**State machine definition**:
```scala
val idle :: read :: read_and_convert :: complete :: Nil = Enum(4)
val state = RegInit(idle)
```

**Key registers**:
```scala
val ConvertBuffer = RegInit(VecInit(Seq.fill(4)(VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W))))))
val rowptr = RegInit(0.U(10.W))    // Convolution window top-left row pointer
val colptr = RegInit(0.U(5.W))     // Convolution window top-left column pointer
val krow_reg = RegInit(0.U(log2Up(b.veclane).W))  // Convolution kernel row count
val kcol_reg = RegInit(0.U(log2Up(b.veclane).W))  // Convolution kernel column count
```

**Command parsing**:
```scala
when(io.cmdReq.fire) {
  rowptr := io.cmdReq.bits.cmd.special(37,28)      // Start row
  colptr := io.cmdReq.bits.cmd.special(27,23)      // Start column
  kcol_reg := io.cmdReq.bits.cmd.special(3,0)      // Convolution kernel column count
  krow_reg := io.cmdReq.bits.cmd.special(7,4)      // Convolution kernel row count
  incol_reg := io.cmdReq.bits.cmd.special(12,8)    // Input matrix column count
  inrow_reg := io.cmdReq.bits.cmd.special(22,13)   // Input matrix row count
}
```

**Data conversion logic**:
```scala
// Fill window data
for (i <- 0 until 4; j <- 0 until 4) {
  when(i.U < krow_reg && j.U < kcol_reg) {
    val bufferRow = (rowcnt + i.U) % krow_reg
    val bufferCol = (colptr + j.U) % incol_reg
    window((i.U * kcol_reg) + j.U) := ConvertBuffer(bufferRow)(bufferCol)
  }.otherwise {
    window((i.U * kcol_reg) + j.U) := 0.U
  }
}
```

**SRAM interface**:
```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)
  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
})
```

**Processing flow**:
1. **idle**: Wait for command, parse convolution parameters
2. **read**: Read initial convolution kernel-sized data into buffer
3. **read_and_convert**: Slide window, convert data and write back
4. **complete**: Send completion signal

**Inputs/Outputs**:
- Input: Ball domain commands containing convolution parameters and address information
- Output: Converted column matrix data, completion signal
- Edge cases: Fill zero values when handling boundaries

## Usage

### Algorithm Principle

**Im2col conversion**: Convert convolution operation to matrix multiplication
- Input: H×W image, K×K convolution kernel
- Output: (H-K+1)×(W-K+1) windows of size K×K, expanded as column vectors

**Sliding window**:
- Slide convolution window in row-major order
- Each window position generates a column vector
- Uses circular buffer to optimize memory access

### Notes

1. **Buffer management**: Uses 4×veclane conversion buffer to store window data
2. **Boundary handling**: Fill zero values for positions beyond image boundaries
3. **Address calculation**: Supports configurable start address and bank selection
4. **Pipeline optimization**: Prefetch next row read requests during conversion
5. **Parameter limitation**: Maximum support for 4×4 convolution kernel size
