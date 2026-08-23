# SMatMulBall

Pebble SMatMulBall contract.

## Instruction

One command computes one C panel already laid out in banks. Compiler slices wide N with extra commands and `op2_base` / `wr_base`.

| Field | Bits | Meaning |
| --- | --- | --- |
| `op1_bank` | `cmd.op1_bank` (`rs1[9:0]`) | A bank |
| `op2_bank` | `cmd.op2_bank` (`rs1[19:10]`) | B bank |
| `wr_bank` | `cmd.wr_bank` (`rs1[29:20]`) | C bank (`cols=4`) |
| `op1_base` | `iter[addrBits-1:0]` | A start row |
| `op2_base` | `iter[2*addrBits-1:addrBits]` | B start row |
| `wr_base` | `iter[3*addrBits-1:2*addrBits]` | C start block |
| `rows` | `rs2[11:0]` | A/C height |
| `cols` | `rs2[23:12]` | B/C width, **1..16** |
| `k` | `rs2[35:24]` | reduction length |

Balls do not read raw `rs1`. Banks come from the decoder; bases are packed in `iter` (`rs1[63:30]`, 34-bit). `addrBits = ceil(log2(bankEntries))` from the chip memdomain (`[bank].entries`). Unused `iter` bits above the three bases must be 0. `rs2[63:36]` must be 0. For pebble `bankEntries=1024`, `addrBits=10`: `op1_base=iter[9:0]`, `op2_base=iter[19:10]`, `wr_base=iter[29:20]`.

The Core assigns the `funct7` encodings for `bb_smatmul_os` and `bb_smatmul_ws` through its `ballISA` configuration. Both use the same `rs1`/`rs2`. Compiler emits WS when `ceil(rows/16) >= 2`, else OS. No cross-instruction ACC.

## Storage

- A/B: `int8`, C: `int32`. Bank row = 128 bit (16×i8 or 4×i32).
- A: M-tile → K-tile; each tile 16×16; pad zeros.
- B: K-tile only (one N panel); each tile 16 rows × `cols` lanes; pad zeros.
- C: block `i` is output row `i` (≤16 i32), striped across `groupBase+0..3` at the same address.
