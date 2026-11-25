# Mxint8actBall UVM Verification

Verifies the generated `Mxint8actBall` module alone with the shared Blink UVM
framework, against a reference ported from the BEMU model
(`emu/src/activation.rs`). The contract is `CONTRACT.md`.

## Structure

- `casegen/src/reference.rs`: MXINT8ACT reference (table lookup, interpolation,
  gate, block exponent selection, rounding and saturation) with counters of the
  semantic paths each case takes
- `casegen/src/cases.rs`: directed and random cases; full bank images of the
  input, table and output banks before and after
- `casegen/src/dpi.rs`: DPI entry points; reads `bank_entries` from
  `BB_VERIFY_CONFIG`
- `src/mxint8act_pkg.sv`: sequence, scoreboard with coverage, env and
  `mxint8act_ball_test`
- `src/tb_top.sv`: `bb_blink_if #(BB_IN_BW, BB_OUT_BW)` + `Mxint8actBall`
  (input and table read ports, one write port)
- `filelists/mxint8act_ball.f`: VCS filelist (`@UVM@` / `@RTL@`)

## Test plan

`mxint8act_ball_test` resets once and runs every case back to back. Each case
gets random distinct bank ids, random group ids and a random ROB id within the
host's widths. Input padding, bytes outside the spans and the output bank hold
random garbage.

| Index | Intent |
|-------|--------|
| 0, 1 | The ctest fixture (M=8, K=32), ungated and gated |
| 2 | Every code at every input exponent from -30 to 10 |
| 3 | Every gate code against swept activation and gate exponents |
| 4 | Extreme and boundary input exponents (-127, -25/-24, -11..-9, 4/5, 127) |
| 5 | Gated scales that clamp the output exponent high or low |
| 6 | Zero, single-lane and +-1 blocks |
| 7 | K=512 (one full scale row) on a random table |
| 8 | K=544 gated (partial second scale row) |
| 9 | The widest row that fits the bank, ending at the bank top |
| 10, 11 | The most rows that fit, ungated and gated |
| 12 | Completion held back by the host for 300 cycles |
| 13 | Interpolation-heavy exponents on a random table, gated |
| 14 | The widest gated row that fits the bank, ending at the bank top |
| 15..38 | Random K, M, gate, bases, exponents, codes and tables |

The scoreboard checks every input read against the input span (twice as long
when gated) and every table read against the 1024-row table span, every write
for routing, full mask, span, single write and data against the reference row,
and after completion the whole output bank (spans and guards) and that the
input and table banks are unchanged.

Coverage, 100% or fatal: K shape (one block, one scale row, full scale row,
several scale rows, widest), partial scale row, M, gated, input span at the
bank top, nonzero bases, held completion, and the semantic paths (tiny
exponent, high and low tails, interior interpolation and table-word crossing,
zero code at a wide exponent, rescale right/exact/left, zero block, output
exponent clamped low/high, rounding tie, code clamp, scale saturation, encode
left shift, flush to zero, zero and negative gates).

Index 4095 of the table cannot be reached by a legal code below the high tail,
so the last-interval path is not a coverage bin. With a 1024-row host the table
base is always 0. Invalid encodings are RTL assertions and are not run here.

## Build / Run

```console
bbdev uvm --run '--ball mxint8act'
```

bbdev elaborates the Ball alone from `configs/default.toml`, builds the
casegen DPI library and passes `MXINT8ACT_FUNCT7`, the Blink widths,
`+UVM_TESTNAME=mxint8act_ball_test` and `+BID=0`.
