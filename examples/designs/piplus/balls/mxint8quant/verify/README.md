# Mxint8quantBall UVM Verification

Verifies the generated `Mxint8quantBall` (MXINT8QUANT: FP32 rows to packed
MXINT8 rows) alone, with the shared BLink UVM framework in `verify/uvm`.

## Structure

- `casegen/`: Rust DPI library. `casegen.rs` holds the reference model, a port
  of `emu/src/quant.rs` (block exponent, round-half-even encode, packed row
  layout, row and column input layouts), and the cases; `dpi.rs` exports them.
- `src/common/mxint8quant_defs.svh`: DPI imports, funct7, timeout
- `src/common/mxint8quant_items.svh`: command item loaded from one Rust step
- `src/seq/mxint8quant_sequences.svh`: one-step sequence
- `src/env/mxint8quant_scoreboard.svh`: preloads the input window, checks reads,
  writes and the response against the reference
- `src/cov/mxint8quant_cov.svh`: semantic covergroups, all required at 100%
- `src/tests/mxint8quant_test.svh`: `mxint8quant_ball_test`, all cases in order
- `src/tb_top.sv`: `bb_blink_if` and the DUT
- `filelists/mxint8quant_ball.f`: VCS filelist (`@UVM@`, `@RTL@`)

## Checks

Each case resets the Ball and runs its steps back to back. For every command
the scoreboard requires that

- each bank row of the input window is read exactly once (rows outside it are
  not preloaded, so the memory model also fails on them), from op1 bank/group;
- each bank row of the packed output is written exactly once, to wr
  bank/group, with a full mask and the reference data (codes, exponent bytes
  and their zero padding), and nothing else is written;
- the response carries the command's rob id and comes after all of it.

Case 0 is `quant_edges_test` and its golden bytes are asserted in the
reference. The other cases cover special blocks (signed zeros, the largest
finite values, subnormal and clamped block maxima, ties, saturation,
flushing), 17-block rows (two exponent rows), 16-block rows, a 4096-value row
filling the bank, column layouts (including columns of a wider matrix and a
bank-filling transpose), 12 random commands back to back, and a response held
40 cycles that must stay valid and stable.

Coverage: K, rows, layout, pitch gap, exponent rows, bases and response hold;
exponent byte and code ranges; reference events (zero block, subnormal block,
clamped exponent, ties down and up, saturation, flush, negative zero,
subnormal element, largest finite maximum).

## Run

```console
nix develop -c bbdev uvm --run '--ball piplus/balls/mxint8quant'
```
