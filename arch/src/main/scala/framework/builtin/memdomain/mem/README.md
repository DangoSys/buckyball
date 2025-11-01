# Memory Bank Implementation

## Overview

Core storage units for BuckyBall's memory domain, located at `arch/src/main/scala/framework/builtin/memdomain/mem`. Provides high-performance on-chip memory components.

Components:
- **SramBank**: Basic SRAM storage bank with synchronous read/write
- **AccBank**: Accumulator storage bank with read-modify-write support
- **Scratchpad**: Scratchpad module managing multiple memory banks with arbitration

## File Structure

```
mem/
├── SramBank.scala     - Basic SRAM bank implementation
├── AccBank.scala      - Accumulator bank implementation
└── Scratchpad.scala   - Scratchpad management module
```

## SramBank.scala

### Interface Definition

```scala
class SramReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class SramWriteReq(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val mask = Vec(mask_len, Bool())
  val data = UInt(w.W)
}
```

### Core Logic

```scala
val mem = SyncReadMem(n, Vec(mask_len, mask_elem))

// Read/write conflict arbitration
assert(!(io.read.req.valid && io.write.req.valid),
       "SramBank: Read and write requests is not allowed at the same time")

io.read.req.ready := !io.write.req.valid
io.write.req.ready := !io.read.req.valid
```

**Constraint**: No simultaneous read/write to same bank in same cycle

## AccBank.scala

### Accumulation Pipeline (AccPipe)

```scala
when (io.write_in.is_acc || RegNext(io.write_in.is_acc)) {
  // Stage 1: Read request
  io.read.req.valid := io.write_in.req.valid

  // Stage 2: Accumulate
  val acc_data = data_reg + io.read.resp.bits.data

  // Stage 3: Write back
  io.write_out.req.bits.data := acc_data
}
```

### Read Request Router (AccReadRouter)

```scala
val req_arbiter = Module(new Arbiter(new SramReadReq(n), 2))
req_arbiter.io.in(0) <> io.read_in2.req  // Higher priority
req_arbiter.io.in(1) <> io.read_in1.req  // Lower priority

// Response distribution
val resp_to_in1 = RegNext(req_arbiter.io.chosen === 1.U && req_arbiter.io.out.fire)
```

## Scratchpad.scala

### Bank Instantiation

```scala
val spad_mems = Seq.fill(sp_banks) { Module(new SramBank(
  spad_bank_entries, spad_w, aligned_to, sp_singleported
)) }

val acc_mems = Seq.fill(acc_banks) { Module(new AccBank(
  acc_bank_entries, acc_w, aligned_to, sp_singleported
)) }
```

### Request Arbitration

```scala
// Read arbitration: priority exec > dma
val exec_read_sel = exec_read_req.valid
val main_read_sel = main_read_req.valid && !exec_read_sel

// Response distribution
val resp_to_main = RegNext(main_read_sel && bank.io.read.req.fire)
val resp_to_exec = RegNext(exec_read_sel && bank.io.read.req.fire)
```

## Important Notes

1. **Single Port Limitation**: Configuration enforces single-port SRAM (`sp_singleported = true`), no same-cycle read/write
2. **Arbitration Priority**: Execution unit (exec) requests have higher priority than DMA requests in all modules
3. **Pipeline Design**: AccBank uses 3-stage pipeline (read-accumulate-write), requires careful data dependency handling
4. **Parameterized Configuration**: All modules support configuration through BaseConfig (bank count, capacity, data width)
5. **Assertions**: Code includes runtime assertions for detecting illegal concurrent access and configuration errors
6. **Mask Support**: Byte-granularity write mask operations, mask length calculated from data width and alignment
