# Memory Domain Modules

Core memory building blocks for Buckyball, located at `arch/src/main/scala/framework/memdomain/mem`. This directory provides single-ported SRAM banks, an accumulation monitor around the bank, and a multi-bank scratchpad with request arbitration.

## Modules

- **SramBank**: Single-ported synchronous SRAM bank with 1-cycle read latency and write-ready arbitration.
- **AccMonitor**: Wraps a `SramBank` with an accumulation pipeline (`AccPipe`) and a read router (`AccReadRouter`). Supports read-modify-write when `is_acc` is asserted.
- **Scratchpad**: Instantiates one `AccMonitor` per bank and arbitrates between DMA and execution-unit (exec) traffic.

## File Layout

```
mem/
├── SramBank.scala    # Basic single-ported SRAM bank
├── AccMonitor.scala  # Accumulate pipeline + read router around SRAM
└── Scratchpad.scala  # Multi-bank manager and request arbitration
```

## Configuration

All modules are parameterized via `examples.BuckyballConfigs.CustomBuckyballConfig` and `org.chipsalliance.cde.config.Parameters`:

- Bank count: `b.sp_banks`, `b.acc_banks`
- Entries per bank: `b.spad_bank_entries`
- Data width: `b.spad_w`
- Alignment: `b.aligned_to`
- Single-ported banks: `b.sp_singleported` (must be true; enforced by assertion in `Scratchpad`)

---

## SramBank

### Interface

```scala
class SramReadReq(val n: Int) extends Bundle {
  val addr    = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class SramReadResp(val w: Int) extends Bundle {
  val data    = UInt(w.W)
  val fromDMA = Bool()
}

class SramWriteReq(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val mask = Vec(mask_len, Bool())
  val data = UInt(w.W)
}
```

### Behavior

- Single-ported arbitration: `assert(!(io.read.req.valid && io.write.req.valid))` forbids same-cycle read+write.
- Handshake: `io.read.req.ready := !io.write.req.valid`, `io.write.req.ready := !io.read.req.valid`.
- Read latency: 1 cycle; `resp.valid := RegNext(io.read.req.fire)`, `resp.bits.data := mem.read(addr, ren).asUInt`.
- Write: stores `data` as a `Vec(mask_len, mask_elem)`. Current implementation writes all lanes; per-bit mask can be added in future.

---

## AccMonitor

`AccMonitor(n, w, aligned_to, single_ported)` composes:

- `SramBank`: underlying single-ported storage.
- `AccPipe`: 3-stage read–accumulate–write pipeline when `is_acc` is asserted.
- `AccReadRouter`: 2-input arbiter for read requests and response distribution.

### AccPipe

- Trigger: pipeline active when `io.write_in.is_acc || RegNext(io.write_in.is_acc)`.
- Stage 1 (Read issue): forwards `addr` and sets `fromDMA := false.B`.
- Stage 2 (Accumulate): if `valid_reg && io.read.io.resp.valid`, compute `acc_data := data_reg + resp.data`; otherwise pass through `data_reg`.
- Stage 3 (Write back): issues write with `acc_data`, preserving `addr` and `mask`.
- Bypass: if not accumulating, write requests pass through directly to SRAM.
- Backpressure: `write_in.ready` mirrors `read.req.ready`; `read.resp.ready` mirrors `write_out.req.ready`.

### AccReadRouter

- Request arbitration: 2-way `Arbiter[SramReadReq]`, with input 0 (second client) prioritized.
- Response routing: captures `req_arbiter.io.chosen` to send `resp` to the initiating client.
- Ready: `read_out.resp.ready` is high when the selected client is ready.
- Safety: `assert(!(io.read_in1.io.req.valid && io.read_in2.io.req.valid))`.

---

## Scratchpad

- Banks: `numBanks = b.sp_banks + b.acc_banks`; instantiates `AccMonitor` per bank.
- Read arbitration per bank: priority `exec` > `dma`.
  - Select: `exec_read_sel = exec_read_req.valid`; `main_read_sel = main_read_req.valid && !exec_read_sel`.
  - Response distribution: records which client fired, then forwards `resp` accordingly.
  - Ready: bank `resp.ready` is the OR of the selected client's ready.
- Write arbitration per bank: priority `exec` > `dma`; muxes `addr`, `data`, `mask` and metadata (`is_acc`, `bank_id`, `rob_id`).
- Metadata: read-side metadata is tied off (`rob_id := 0.U`, `is_acc := false.B`, `bank_id := i.U`).
- Assertions:
  - `assert(b.sp_singleported)` — Scratchpad expects single-ported SRAM.
  - `assert(!(exec_read_req.valid && exec_write.io.req.valid))` — exec read and write cannot target the same bank simultaneously.

---

## Key Guarantees & Notes

1. **Single-Port Discipline**: No same-cycle read/write in `SramBank`; higher layers arbitrate accordingly.
2. **Priority Policy**: `exec` traffic wins over `dma` for both reads and writes; `AccReadRouter` prioritizes its second input.
3. **Accumulation Semantics**: When `is_acc` is asserted, writes become read-modify-write with `+` over the fetched word.
4. **Masking**: Write requests carry a mask vector; current `SramBank` implementation writes all lanes and can be extended to honor per-bit masks.
5. **Latency**: Reads have 1-cycle latency from `req.fire` to `resp.valid`; accumulation path adds pipeline staging.
6. **Parameterization**: Widths, depths, and counts come from config; ensure `w` and `aligned_to` satisfy `require(w % aligned_to == 0 || w < aligned_to)`.

---

## Where to Look

- Bank implementation: `SramBank.scala`
- Accumulate pipeline & router: `AccMonitor.scala`
- Multi-bank arbitration & top-level wiring: `Scratchpad.scala`

These modules are designed to be composed cleanly with Chisel handshakes and explicit assertions to catch illegal concurrent accesses.
