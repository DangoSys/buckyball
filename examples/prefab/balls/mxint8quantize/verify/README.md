# Mxint8quantizeBall UVM Verification

Verifies the generated `Mxint8quantizeBall` (with its 16 `Encode` modules)
alone, on the shared BLink UVM framework in `verify/uvm/src/ball`.

## Structure

- `casegen/`: Rust DPI library (`mxint8quantize-casegen`). `casegen.rs` is the
  reference model, ported from `emu/src/quant.rs`, and builds every case:
  preload rows, expected read addresses, expected writes (address, mask,
  data) and per-step counts of numeric events for coverage.
- `src/common/`: DPI imports, the command item loading one step from Rust.
- `src/seq/`: one-step sequence.
- `src/env/`: scoreboard and env (extends `bb_blink_env`).
- `src/cov/`: command, exponent byte, code and numeric event covergroups;
  `check_phase` fatals below 100%.
- `src/tests/`: `mxint8quantize_ball_test`, runs every case.
- `src/tb_top.sv`: `bb_blink_if` + `Mxint8quantizeBall`. firtool folds the
  constant all-true write mask out of the port list, so the testbench drives
  the interface mask with `16'hffff`.
- `filelists/mxint8quantize_ball.f`: VCS filelist (`@UVM@`, `@RTL@`).

## Checks

The scoreboard compares, per command, the observed command with the
stimulus, every read (port, bank, group, ROB id, address, in order: the
whole input window and nothing else), every write (bank, group, ROB id,
address, mask and data, in order: two code rows per block, the zero-padded
exponent row at every 16th and each row's last block) and the response
(ROB id, after all traffic). The memory model fatals on reads outside the
preloaded input window.

## Cases

| case | what                                                                 |
|------|----------------------------------------------------------------------|
| 0    | `quant_edges_test`: ties to even, clamp at 127, signed zeros, all-zero block; the reference is asserted against its hand-derived bytes |
| 1    | subnormal block maximum, maxima F = 1..6 (exponent clamps to -127), F = 7, signed zeros only, flush of tiny values, F = 254, -FLT_MAX |
| 2    | exponent sweep over the full input bank: 2 rows of K = 2048, block maxima F = 1, 3, .. 253, 254 |
| 3    | largest K (4096, one row fills the input), output ends at the last bank row |
| 4    | most rows (128) of K = 32, input fills the bank, output ends at the last row |
| 5    | `quant_random_test` shape: 3 rows of K = 544 (tail exponent row), nonzero bases |
| 6    | exact half steps and saturation, 2 rows of K = 512                    |
| 7    | 4 back-to-back commands: bank ids 0 and 1023, groups 0 and 23, ROB ids 0..7 |
| 8    | 16 random back-to-back commands: K, rows, bases, banks, groups, ROB ids, scales, zeros |

## Run

```console
bbdev uvm --run '--ball mxint8quantize'
```

The Ball runs alone from `configs/default.toml` (ball id 0, `+BID=0`); bank
entries come from `BB_VERIFY_CONFIG`.
