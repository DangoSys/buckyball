# MatrixBall

Pebble MatrixBall contract (fail-hard). Stack order: ctest → bemu → compiler → mlir_tests → RTL → UVM.

## Instruction (funct7 = 65)

| Field | Bits | Meaning |
| --- | --- | --- |
| `op1_bank` | `rs1[9:0]` | A bank |
| `op2_bank` | `rs1[19:10]` | B bank |
| `wr_bank` | `rs1[29:20]` | C bank (`cols=4`) |
| `op1_base` | `rs1[36:30]` | A start row |
| `op2_base` | `rs1[43:37]` | B start row |
| `wr_base` | `rs1[50:44]` | C start block |
| `M` | `rs2[11:0]` | rows of A/C |
| `N` | `rs2[23:12]` | cols of B/C |
| `K` | `rs2[35:24]` | cols of A / rows of B |
| `mode` | `rs2[36]` | `0` OS, `1` WS |

C macros: `bb_matrix_mnk_mode`, `bb_matrix_mnk`, `bb_matrix_os`, `bb_matrix_ws`.
No cross-instruction ACC phases.

## Storage

- A/B: `int8`, C: `int32`. Bank row = 128 bit (16×i8 or 4×i32).
- A: M-tile → K-tile; each tile 16×16; pad zeros.
- B: N-tile → K-tile; each tile 16×16; pad zeros.
- C: each block is one row of one N-tile (≤16 i32), striped across `groupBase+0..3` at the same address. Final layout matches OS M-tile / N-tile / row order (WS remaps to the same addresses).

## CTests

Under `workloads/ctests/`. Pack helpers in `matrix_test_common.h`. Every test compares CPU golden.

- Small tests: hand-written / deterministic inputs; in bemu and Verilator regression.
- Bank tests (`matrix_bank_*`, ≥1024 A rows): random inputs; bemu regression only (not Verilator).
