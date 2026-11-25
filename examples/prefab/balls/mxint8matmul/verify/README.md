# Mxint8matmulBall UVM Verification

Verifies `Mxint8matmulBall` alone (its BallDomain from `configs/default.toml`)
on the shared BLink UVM framework in `verify/uvm`.

```console
nix develop -c bbdev uvm --run '--ball mxint8matmul'
```

## Layout

- `casegen/`: Rust DPI library (`mxint8matmul-casegen`)
  - `reference.rs`: the MXINT8MM reference, ported from `emu/src/matmul.rs`
    (block sums, RNE conversion of each block term including subnormals,
    block-ordered FP32 accumulation onto +0.0 or the old C), and the semantic
    events of each command for coverage
  - `cases.rs`: directed and random cases; a case is a bank image plus
    commands run back to back without reset
  - `dpi.rs`: case loading, bank image, commands and expected C rows
- `src/mxint8matmul_pkg.sv`: sequence, scoreboard with covergroups, env, test
- `src/tb_top.sv`: `bb_blink_if` and the `Mxint8matmulBall` DUT
- `filelists/mxint8matmul_ball.f`: VCS filelist (`@UVM@`, `@RTL@`)
- `parameters.toml`: the PE array `rows` and `cols` for the shape bins

## Cases

| case | what |
|------|------|
| 0 | 1x4, K = 32, all bases 0 |
| 1 | 5x12, K = 64: partial row and column tiles, bases 1/2/3 |
| 2 | accumulate onto an old C in a renamed bank |
| 3 | accumulate onto the destination bank itself |
| 4 | K = 544 (two exponent rows), bases 3/1/2, then accumulated in place |
| 5 | K = 512 (one full exponent row), then accumulated into a renamed bank |
| 6 | K = 1024, then accumulated in place |
| 7 | subnormal terms: rounding, ties to even both ways, -0.0 terms |
| 8 | exponent bytes 0 flush every term; old C -0.0, +0.0, subnormals |
| 9 | extreme products 2^19 cancelled exactly in the next block |
| 10 | large terms from exponent bytes 255 and 177 |
| 11 | A, B and C windows end at the last bank row |
| 12 | M = 256, N = 4 (many row tiles) |
| 13 | N = 128 (many column tiles) |
| 14 | K = 96 split over three commands (overwrite, renamed, in place); casegen asserts it equals one K = 96 command |
| 15 | four independent commands back to back |
| 16..39 | random M, N, K, bases, banks, groups, codes, exponents, old C |

Case `i` holds every response back for `(i % 3) * 20` cycles; the response
must stay stable and the Ball must not take a new command meanwhile. ROB ids
count up and wrap at `BB_ROB_ID_W`.

## Checks

- accepted command equals the driven one
- reads: port 0 only in the A window or (when accumulating) the old C window
  of `wr_read_bank`, port 1 only in the B window; every old C row is read
- writes: port 0, `wr_bank`/`wr_group`, full mask, inside the C window, each
  row once, data bit-exact against the reference; the response comes after all
  C rows are written and carries the command's ROB id
- `check_phase` fails unless both covergroups reach 100%

## Coverage

- `command_cg`: M and N against the PE array (below one tile, one tile,
  several tiles with a tail, whole tiles) and their cross; K (32, 64..480,
  512, 544..992, 1024+) crossed with overwrite/accumulate; old C in place or
  renamed; windows ending at the last bank row; zero or offset bases; every
  ROB id
- `feature_cg`: subnormal, rounded, tie-down, tie-up and -0.0 terms, terms of
  2^64 and above, products of 2^19, inexact and cancelling sums, -0.0 and
  subnormal results, code -128, exponent bytes 0 and >= 128, old C -0.0 or
  subnormal

`cargo test` in `casegen` checks that the cases hit every semantic event.
