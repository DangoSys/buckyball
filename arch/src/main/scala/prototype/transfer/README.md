# Transfer Tile Data Mover Accelerator

## Overview

This directory implements Buckyball's tile-based Transfer accelerator, located at `arch/src/main/scala/prototype/transfer`. The module copies Scratchpad data by tiles (`veclane × veclane`) from a source bank/address to a destination bank/address, in a vectorized manner, and writes results back without any arithmetic modification.

Core components:
- **Transfer.scala**: Transfer accelerator main implementation

## Code Structure

```
transfer/
└── Transfer.scala  - Transfer accelerator implementation
```

### Module Responsibilities

**Transfer.scala** (Accelerator implementation layer)
- Reads a `veclane × veclane` tile of data from a source Scratchpad bank/address
- Performs no element-wise computation (pure copy)
- Packs rows and writes back the same-sized tile with full mask to the destination bank/address
- Provides Ball domain command interface and returns completion response/status

## Module Description

### Transfer.scala

**Main functionality**:

Read input tile by tile (`veclane × veclane`) → Pack row data → Write output back row by row; supports `iter`-driven batch processing and pipelined workflow.

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
  respCounter  := 0.U      // Clear read response row count
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

**Data path (split & pack)**:

- Read returns a packed data row of width `spad_w = veclane × inputWidth`;
- Split into `veclane` elements (no arithmetic modification) and buffer into `regArray` by row;
- When writing back, repack a full row of `veclane` elements:

```scala
// Split by column
val dataWord = io.sramRead(rbank_reg).resp.bits.data
for (col <- 0 until b.veclane) {
  val hi = (col + 1) * b.inputType.getWidth - 1
  val lo = col * b.inputType.getWidth
  val raw = dataWord(hi, lo)
  regArray(respCounter)(col) := raw.asUInt
}

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

1. **idle**: Wait for command; parse input/output bank/addr, iteration count `iter`, and calculate `cycle_reg` accordingly. Optionally, advance to next round by adjusting addresses when `cycle_reg =/= 0`.
2. **sRead**: Issue consecutive read requests row by row for the source bank/addr; after receiving data, split and fill into `regArray`; enter write phase after accumulating `veclane` rows.
3. **sWrite**: Pack and write back row by row to consecutive addresses in `wbank` with full mask write; enter complete phase after writing `veclane` rows.
4. **complete**: When all rounds complete (`cycle_reg == 0`), issue `cmdResp` completion response; then return to `idle`.

**Inputs/Outputs**:

- Input: Ball domain commands (`wr_bank/wr_bank_addr`, `op1_bank/op1_bank_addr`, `iter`, etc.)
- Output: Copied tile written to destination bank, `cmdResp` completion notification
- Boundaries and constraints:
  - Each round processes `veclane` rows, iteration round count derived from `iter`;
  - Write-back uses full mask (can be extended for partial write as needed).

## ISA Structure

The Ball instruction corresponding to this module performs tile-based copy on Scratchpad data and writes back to the destination.

**Function**: Copy input rows from source Scratchpad address to destination Scratchpad address, write back row by row, loop `iter` times.

**Format**: `bb_transfer rs1, rs2`

**Operands**:

- `rs1[spAddrLen-1:0]`: Source Scratchpad address (op1_spaddr)
- `rs2[spAddrLen-1:0]`: Destination Scratchpad address (wr_spaddr)
- `rs2[spAddrLen+9:spAddrLen]`: Iteration count (iter, row count)
- `rs2[63:spAddrLen+10]`: special/reserved field (not used by current Transfer)

Address note: Local address of `spAddrLen` width is further split into bank and row in hardware (see LocalAddr), no need to explicitly distinguish at ISA level.

**Operation**: Read data from Scratchpad address specified by `rs1`, then write back row by row to address specified by `rs2`, loop `iter` times.

rs1 (source address):

```
┌──────────────────────────────────────────────────────┐
│                   op1_spaddr                         │
│                 (spAddrLen bits)                     │
├──────────────────────────────────────────────────────┤
│                 [spAddrLen-1:0]                      │
└──────────────────────────────────────────────────────┘
```

rs2 (destination address and iteration count):

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
- Configure destination `wr_bank/wr_bank_addr`, and element row count `iter` to process;
- After sending Ball command, wait for `cmdResp` completion;
- Can poll `status`: `ready/valid/idle/init/running/complete/iter` to get runtime information.

### Notes

1. **Handshake robustness**: Prefer gating `resp.ready` by local buffer availability (e.g., `respCounter < readCounter && respCounter < b.veclane.U`) to avoid overruns; simple `true.B` is acceptable when the Scratchpad controller guarantees pacing.
2. **Bandwidth and alignment**: Each read/write is one packed row (`spad_w` bits), addresses need to be row-aligned and increment consecutively.
3. **Mask strategy**: Current implementation uses full mask write; if sparse/partial write needed, extend `writeMaskReg` generation logic or use constant full-ones without a register.
4. **Iteration and chunking**: When `iter` is not a multiple of `veclane`, `cycle_reg` handles remaining rows with ceiling rounding; addresses advance by `veclane` per round.
5. **Read/Write phasing**: If the Scratchpad bank ports disallow same-bank concurrent read/write, keep two-phase read-then-write; otherwise consider streaming per-row read→write to reduce buffer size.
6. **Reset behavior**: Reset clears `regArray`, `writeDataReg`, `writeMaskReg`, facilitating reproducible simulation.
