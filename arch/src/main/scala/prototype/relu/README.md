# ReLU Activation Function Accelerator

## Overview

This directory implements BuckyBall's ReLU (Rectified Linear Unit) activation accelerator, located at `arch/src/main/scala/prototype/relu`. The module performs element-wise ReLU processing on Scratchpad data by tiles (`veclane × veclane`) in a vectorized manner and writes results back.

Core components:
- **Relu.scala**: ReLU accelerator main implementation

## Code Structure

```
relu/
└── Relu.scala  - ReLU accelerator implementation
```

### Module Responsibilities

**Relu.scala** (Accelerator implementation layer)
- Reads a `veclane × veclane` tile of data from Scratchpad
- Performs signed comparison-based ReLU operation on each element (negative values set to 0)
- Writes back same-sized tile with full mask write
- Provides Ball domain command interface and returns completion response/status

## Module Description

### Relu.scala

**Main functionality**:

Read input tile by tile (`veclane × veclane`) → Execute element-wise ReLU → Write output back row by row; supports `iter`-driven batch processing and pipelined workflow.

**State machine definition**:

```scala
val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
val state = RegInit(idle)
```

**Key registers**:

```scala
// Data cache: veclane × veclane, each element width is inputType.getWidth
val regArray = RegInit(
  VecInit(Seq.fill(b.veclane)(
    VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
  ))
)

// Counters
val readCounter  = RegInit(0.U(log2Ceil(b.veclane + 1).W)) // Read request row count
val respCounter  = RegInit(0.U(log2Ceil(b.veclane + 1).W)) // Read response row count
val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W)) // Write-back row count

// Instruction field registers
val robid_reg = RegInit(0.U(10.W)) // Command ROB ID
val waddr_reg = RegInit(0.U(10.W)) // Write-back start row address
val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W)) // Write-back target Scratchpad bank selection
val raddr_reg = RegInit(0.U(10.W)) // Read start row address
val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W)) // Read source Scratchpad bank selection
val iter_reg  = RegInit(0.U(10.W)) // Processing row count/length specified in command
val cycle_reg = RegInit(0.U(6.W))      // Tile round count (derived from iter)
val iterCnt   = RegInit(0.U(32.W))     // Completed batch count

// Write-back data and mask
val spad_w       = b.veclane * b.inputType.getWidth // Packed data width per row
val writeDataReg = Reg(UInt(spad_w.W)) // Packed data to write back per row
val writeMaskReg = Reg(Vec(b.spad_mask_len, UInt(1.W))) // Write-back mask vector
```

**Command parsing**:

```scala
when(io.cmdReq.fire) {
  // Enter read phase and initialize round counters
  state        := sRead
  readCounter  := 0.U      // Clear read request row count
  writeCounter := 0.U      // Clear write-back row count

  // Record command identifier
  robid_reg := io.cmdReq.bits.rob_id            // ROB ID (for completion response matching)

  // Output (write-back) target address: use wr_* fields
  waddr_reg := io.cmdReq.bits.cmd.wr_bank_addr  // Write-back start row address
  wbank_reg := io.cmdReq.bits.cmd.wr_bank       // Write-back target bank

  // Input (read) source address: use op1_* fields
  raddr_reg := io.cmdReq.bits.cmd.op1_bank_addr // Read start row address
  rbank_reg := io.cmdReq.bits.cmd.op1_bank      // Read source bank

  // Iteration and rounds
  iter_reg  := io.cmdReq.bits.cmd.iter          // Total rows to process (iteration count)
  // Calculate required tile rounds for this batch: each round processes veclane rows
  // cycle_reg = ceil(iter / veclane) - 1, decrements after each read/write round completes
  cycle_reg := (io.cmdReq.bits.cmd.iter +& (b.veclane.U - 1.U)) / b.veclane.U - 1.U
}
```

**Data conversion logic (ReLU)**:

- Read returns a packed data row of width `spad_w = veclane × inputWidth`;
- Split into `veclane` elements and perform signed comparison: `x < 0 ? 0 : x`;

```scala
// Split + ReLU (by column)
val dataWord = io.sramRead(rbank_reg).resp.bits.data
for (col <- 0 until b.veclane) {
  val hi = (col + 1) * b.inputType.getWidth - 1
  val lo = col * b.inputType.getWidth
  val raw    = dataWord(hi, lo)
  val signed = raw.asSInt
  val relu   = Mux(signed < 0.S, 0.S(b.inputType.getWidth.W), signed)
  regArray(respCounter)(col) := relu.asUInt
}
```

When writing back, repack a full row of `veclane` elements:

```scala
// Pack regArray(rowIdx) into one row for write-back
writeDataReg := Cat((0 until b.veclane).reverse.map(j => regArray(rowIdx)(j)))
// Full mask write
for (i <- 0 until b.spad_mask_len) { writeMaskReg(i) := 1.U }
```

**SRAM interface**:

```scala
val io = IO(new Bundle {
  val cmdReq    = Flipped(Decoupled(new BallRsIssue))
  val cmdResp   = Decoupled(new BallRsComplete)
  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
  val status    = new Status
})
```

**Processing flow**:

1. **idle**: Wait for command; parse input/output address bank/addr, iteration count `iter`, and calculate `cycle_reg` accordingly.
2. **sRead**: Issue consecutive read requests row by row for input bank/addr; after receiving data, perform element-wise ReLU and fill into `regArray`; enter write phase after accumulating `veclane` rows.
3. **sWrite**: Pack and write back row by row to consecutive addresses in `wbank` with full mask write; enter complete phase after writing `veclane` rows.
4. **complete**: When all rounds complete (`cycle_reg == 0`), issue `cmdResp` completion response; then return to `idle`.

**Inputs/Outputs**:

- Input: Ball domain commands (`wr_bank/wr_bank_addr`, `op1_bank/op1_bank_addr`, `iter`, etc.)
- Output: ReLU result tile after write-back, `cmdResp` completion notification
- Boundaries and constraints:
  - Each round processes `veclane` rows, iteration round count derived from `iter`;
  - Data elements perform signed comparison based on `b.inputType`;
  - Write-back uses full mask (can be extended for partial write as needed).

## ISA Structure

The Ball instruction corresponding to this module performs element-wise ReLU on Scratchpad data and writes back.

**Function**: Execute element-wise ReLU on input matrix (negative values set to 0), write back row by row to target address.

**func7**: `0100110` 38 (corresponds to `DISA.RELU`)

**Format**: `bb_relu rs1, rs2`

**Operands**:

- `rs1[spAddrLen-1:0]`: Source operand Scratchpad address (op1_spaddr)
- `rs2[spAddrLen-1:0]`: Result write-back Scratchpad address (wr_spaddr)
- `rs2[spAddrLen+9:spAddrLen]`: Iteration count (iter, row count)
- `rs2[63:spAddrLen+10]`: special/reserved field (not used by current ReLU)

Address note: Local address of `spAddrLen` width is further split into bank and row in hardware (see LocalAddr), no need to explicitly distinguish at ISA level.

**Operation**: Read data from Scratchpad address specified by `rs1`, perform element-wise ReLU, then write results back row by row to address specified by `rs2`, loop `iter` times.

rs1 (input address):

```
┌──────────────────────────────────────────────────────┐
│                   op1_spaddr                         │
│                 (spAddrLen bits)                     │
├──────────────────────────────────────────────────────┤
│                 [spAddrLen-1:0]                      │
└──────────────────────────────────────────────────────┘
```

rs2 (write-back address and iteration count):

```
┌──────────────────────────────────┬────────────────────┐
│        iter (rows)               │    wr_spaddr       │
│        (10 bits)                 │ (spAddrLen bits)   │
├──────────────────────────────────┼────────────────────┤
│ [spAddrLen+9: spAddrLen]         │  [spAddrLen-1:0]   │
└──────────────────────────────────┴────────────────────┘
```

Note: During decode, `op1_spaddr` comes from `rs1`, `wr_spaddr` and `iter` come from `rs2`, remaining `special` high bits can be reserved for extension.

## Usage

- Place source data at Scratchpad starting position specified by `op1_bank/op1_bank_addr`, ensure each row width is `veclane × inputWidth`;
- Configure output `wr_bank/wr_bank_addr`, and element row count `iter` to process;
- After sending Ball command, wait for `cmdResp` completion;
- Can poll `status`: `ready/valid/idle/init/running/complete/iter` to get runtime information.

### Notes

1. **Signed comparison**: ReLU uses `asSInt` for negative value detection, negative values set to 0; ensure `b.inputType` matches upstream data convention (fixed-point/two's complement).
2. **Bandwidth and alignment**: Each read/write is one packed row (`spad_w` bits), addresses need to be row-aligned and increment consecutively.
3. **Mask strategy**: Current implementation uses full mask write; if sparse/partial write needed, extend `writeMaskReg` generation logic.
4. **Iteration and chunking**: When `iter` is not a multiple of `veclane`, `cycle_reg` handles remaining rows with ceiling rounding; pad 0 or trim at boundaries if necessary.
5. **Submodule interaction**: `sramRead/Write` ready/valid handshake needs to be consistent with Scratchpad controller; if out-of-order/multi-cycle delay exists, protect out-of-order cases in `respCounter` logic.
6. **Reset behavior**: Reset clears `regArray`, `writeDataReg`, `writeMaskReg`, facilitating reproducible simulation.
