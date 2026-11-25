# Mxint8gemvBall UVM Verification

Verifies the generated `Mxint8gemvBall` module against the integer reference of
`CONTRACT.md` (a port of `emu/src/gemv.rs`), using the shared Blink UVM framework.

## Structure

- `casegen/src/reference.rs`: one MXINT8GEMV instruction on bank images
  (INT32 block dots, shared-scale INT64 state with signed RNE, finalize to
  MXINT8); records which numerical paths a case took
- `casegen/src/cases.rs`: directed and random cases and the DPI API
  (`mxint8_case_count/load/field/word`)
- `src/mxint8gemv_pkg.sv`: sequence, scoreboard, contract covergroup, env, test
- `src/tb_top.sv`: `bb_blink_if` + `Mxint8gemvBall`, 16 KiB bank bound checks
- `filelists/mxint8gemv_ball.f`: VCS filelist

## Cases

One test, `mxint8gemv_ball_test`, runs 58 cases, each after a reset:

- 0/1: K32 partial/final, cross-group channel order, nonzero bases
- 2/3: K512 partial (C base 0), K1024 final (A/B exponent row refresh, per-block exponent
  jumps); case 3 holds the completion under response backpressure
- 4/5: accumulate on a valid state (K64 partial; K512 final with in-place C)
- 6/7: all-zero blocks; finalize of a zero accumulated state (exponent 127)
- 8: exact cancellation clears the scale, the next block selects a fresh one
- 9: scale -254 -> 254 -> -254 (state and term right shifts of at least 64)
- 10: odd products shifted by one and two (signed ties to even)
- 11/12/13: finalize with q clamped high (saturation, exact left shift) and low
- 14: K1984 with all codes -128, every window at the top of its bank
- 15/16: accumulated total K of exactly 32768 (partial and final)
- 17: finalize ties: +-127.5 rounds to +-128 and clamps to +-127
- 18..57: random K, bases, banks/groups, ROB ids, code and exponent
  distributions, accumulate/finalize/in-place

All A/B/old-C rows outside the operand windows hold random bytes.

## Scoreboard

- every read on its contract channel (0: A codes/exponents or old C; 1+g: B
  group g) and inside its window; old C is read 17 rows and before any write
- exactly 17 (partial) or 3 (final) writes, in row order from the C base, full
  mask, data equal to the reference
- at completion the whole destination bank equals the reference image and the
  A, B and old C images are unchanged

`contract_cov` (K classes, accumulate x finalize, base placement, and one bin per
numerical path: realign, flush, term rounding, ties, zero blocks, restart, q
clamps, saturation, left shift, zero output, -128 codes, K limit, valid/zero old
state, in-place C) must reach 100%, else `check_phase` fatals.

Invalid fields trip RTL assertions, which end the simulation; they are not
covered here.

## Run

```console
bbdev uvm --run '--ball mxint8gemv'
```

The Ball runs alone from `configs/default.toml` (id 0, `+BID=0`).
