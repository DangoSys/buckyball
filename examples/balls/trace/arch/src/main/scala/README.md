# TraceBall — debug/trace Ball

## Overview

TraceBall is a non-computing Ball that exposes runtime debug features through the Buckyball instruction channel. It has two main roles:

1. **Cycle counter** — set/release multiple independent cycle counters via instructions to measure elapsed cycles over arbitrary code regions
2. **Bank backdoor** — inject data into SRAM banks or read bank data out via DPI-C

All DPI-C hooks live inside TraceBall only and do not affect other modules.

---

## Instruction encoding

TraceBall uses **two funct7 encodings**.

### Instruction 1: `bdb_counter` (funct7 = 48, 0x30)

Cycle counter control. **Does not touch SRAM, needs no bank ports, completes in 1 cycle.**

rs1 layout:
- rs1 = don’t care (unused; no need to set BB_RD0/BB_WR)

rs2 layout (64-bit):
```
rs2[3:0]   = subcmd    subcommand (0=START, 1=STOP, 2=READ)
rs2[7:4]   = ctr_id    counter id (0–15, up to 16 independent counters)
rs2[63:8]  = payload
```

Subcommands:

| subcmd | name | behavior | payload meaning |
|--------|------|----------|-----------------|
| 0 | `CTR_START` | Start counter `ctr_id`, record current cycle as start | payload = tag (user tag, echoed in trace) |
| 1 | `CTR_STOP` | Stop `ctr_id`, emit elapsed cycles to DPI-C trace, then free the counter | payload = ignored |
| 2 | `CTR_READ` | Read current value of `ctr_id` (does not stop), emit to DPI-C trace | payload = ignored |

DPI-C trace format (written to bdb.log):
```
[CTRACE] CTR_START  ctr=0 tag=0xDEAD cycle=10042
[CTRACE] CTR_STOP   ctr=0 tag=0xDEAD elapsed=387 cycle=10429
[CTRACE] CTR_READ   ctr=0 current=200 cycle=10242
```

### Instruction 2: `bdb_backdoor` (funct7 = 49, 0x31)

SRAM backdoor read/write; **all parameters (bank_id, row, data) come from DPI-C**. **Requires bank ports (inBW=1, outBW=1).**

rs1 layout:
```
rs1[45]     = BB_RD0    read: get address from DPI-C, read SRAM, send data to DPI-C
rs1[47]     = BB_WR     write: get address+data from DPI-C, write SRAM
rs1[63:48]  = iter      repeat count (0 = once, >0 = loop `iter` times)
```

rs2 layout:
- rs2 = don’t care (unused)

Modes:

| rs1 flag | behavior | DPI-C interaction |
|----------|----------|-------------------|
| BB_RD0 | read | RTL calls `dpi_backdoor_get_read_addr()` for (bank_id, row), reads SRAM, `dpi_backdoor_put_read_data()` for data |
| BB_WR | write | RTL calls `dpi_backdoor_get_write_req()` for (bank_id, row, data), writes SRAM |

DPI-C trace format:
```
[BANK-TRACE] BACKDOOR_READ  bank=2 row=5 data=0x00010002000300040005000600070008
[BANK-TRACE] BACKDOOR_WRITE bank=3 row=10 data=0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF
```


## Examples

### Measure kernel cycles

```c
// One matmul region
bdb_counter_start(0, 0xA001);         // counter 0, tag=matmul
bb_mul_warp16(A, B, C, 16);
bb_fence();
bdb_counter_stop(0);                  // prints elapsed

// Nested regions
bdb_counter_start(0, 0xB001);         // outer: whole conv
  bdb_counter_start(1, 0xB002);       // inner: im2col
  bb_im2col(...);
  bb_fence();
  bdb_counter_stop(1);

  bdb_counter_start(2, 0xB003);       // inner: matmul
  bb_mul_warp16(...);
  bb_fence();
  bdb_counter_stop(2);
bdb_counter_stop(0);                    // outer end
```

bdb.log sample:
```
[CTRACE] CTR_START  ctr=0 tag=0xB001 cycle=0
[CTRACE] CTR_START  ctr=1 tag=0xB002 cycle=0
[CTRACE] CTR_STOP   ctr=1 tag=0xB002 elapsed=150 cycle=150
[CTRACE] CTR_START  ctr=2 tag=0xB003 cycle=0
[CTRACE] CTR_STOP   ctr=2 tag=0xB003 elapsed=300 cycle=300
[CTRACE] CTR_STOP   ctr=0 tag=0xB001 elapsed=456 cycle=456
```

### Backdoor test data into SRAM

TraceBall has a private bank that is not visible to normal configuration.

```c
// No DMA: inject test data into bank 0 via DPI-C
// (C++ first injects into TraceBall’s internal bank via DPI-C)
bb_alloc(0, 1, 1);
bdb_backdoor_mvin(16);            // inject 16 rows into private bank
bdb_backdoor_write(0, 16);        // move 16 rows from private bank to bank 0

bb_transpose(0, 1, 16);
bdb_backdoor_peek(0, 15);         // inspect last row

bdb_backdoor_read(1, 16);         // dump 16 rows from bank 1 to trace
```



All RTL changes stay inside TraceBall; do not modify external banks or shared infrastructure there.
