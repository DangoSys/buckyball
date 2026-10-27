# TraceBall — debug/trace Ball

## Overview

TraceBall is a non-computing Ball that measures elapsed cycles over arbitrary
code regions with up to sixteen independent cycle counters. It touches no bank.
The DPI-C hook lives inside TraceBall only and does not affect other modules.

The framework requires every Ball to own at least one read and one write bank
port, so the design config gives TraceBall `inBW = 1, outBW = 1`; TraceBall
ties those ports idle and never issues a bank request.

---

## Instruction: `bdb_counter`

The selected Core's `ballISA` assigns the `funct7` of `BDB_COUNTER`.
Cycle counter control; completes in one cycle after acceptance.

rs1 is unused and zero.

rs2 layout (64-bit):
```
rs2[3:0]   = subcmd    subcommand (0=START, 1=STOP, 2=READ; 3..15 are illegal)
rs2[7:4]   = ctr_id    counter id (0–15)
rs2[63:8]  = payload
```

| subcmd | name | behavior | payload |
|--------|------|----------|---------|
| 0 | `CTR_START` | Start counter `ctr_id`, record the current cycle and the tag | tag, echoed in the trace |
| 1 | `CTR_STOP` | Emit elapsed cycles of an active `ctr_id`, then free it | ignored |
| 2 | `CTR_READ` | Emit elapsed cycles of an active `ctr_id`; it keeps running | ignored |

Illegal inputs stop the simulation: an RTL assertion and a BEMU panic.
- `subcmd` 3..15.
- `CTR_STOP` or `CTR_READ` of a counter that is not started.

## Counter trace

With `--ctrace`, both Verilator (`dpi_ctrace`) and BEMU append one NDJSON
record per command to `bdb.ndjson`, so the same tools read both:
```
{"type":"ctrace","clk":10042,"event":"ctr_start","ctr_id":0,"tag":"0xDEAD","cycle":10042}
{"type":"ctrace","clk":10429,"event":"ctr_stop","ctr_id":0,"tag":"0xDEAD","elapsed":387,"cycle":10429}
{"type":"ctrace","clk":10242,"event":"ctr_read","ctr_id":0,"current":200,"cycle":10242}
```
`cycle` is the Ball's cycle count and `elapsed`/`current` is `cycle` minus the
cycle of the counter's START. `clk` is the simulator clock. BEMU uses the issue
cycle of the instruction for both `clk` and `cycle`.

## Example

```c
bdb_counter_start(0, 0xB001);         // outer: whole conv
  bdb_counter_start(1, 0xB002);       // inner: im2col
  bb_im2col(...);
  bdb_counter_stop(1);

  bdb_counter_start(2, 0xB003);       // inner: matmul
  bb_vecmat16(...);
  bdb_counter_stop(2);
bdb_counter_stop(0);
```
