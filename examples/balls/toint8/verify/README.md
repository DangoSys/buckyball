# ToInt8Ball UVM Verification

Verifies the generated `ToInt8Ball` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/toint8_defs.svh`: DPI imports, `toint8_require_bid`, timeouts
- `src/common/toint8_items.svh`: `toint8_cmd_item` with `load_rust_case` (Da address in `special[12:0]`)
- `src/seq/toint8_sequences.svh`: one-case sequence
- `src/cov/toint8_cov.svh`: online INT8 layout and iter 1 coverage
- `src/env/toint8_scoreboard.svh`: preload mem model (group-aware for i8), compare writes vs DPI ref
- `src/env/toint8_env.svh`: extends `bb_blink_env#(1,1)`
- `src/tests/toint8_*_test.svh`: one directed case per UVM test
- `src/pkg/toint8_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(1,1)` + `ToInt8Ball`
- `filelists/toint8_ball.f`: VCS filelist

## Test plan

The test names are:

- `toint8_signed_test`: signed values across all four source groups
- `toint8_zero_test`: all-zero activation (`Da=1.0`)
- `toint8_rounding_test`: ties-to-even quantization boundaries
- `toint8_rows_test`: two-row scan and packed output traversal
- `toint8_scale_rows_test`: maximum value is in the second row

Each case: reset, preload src words into `mem_model` (INT8 uses `group_id` 0..3),
drive one command, wait for `scb.done()` or timeout at 400 cycles. Scoreboard
compares observed writes with DPI `toint8_ref_i8`. `toint8_cov` requires the online
INT8 layout and one-row iteration bin, else `check_phase` fatal.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=3` |
| `sims.verilator.BuckyballToyVerilatorConfig` (full.toml) | `+BID=5` |

```console
./simv +UVM_TESTNAME=toint8_scale_rows_test +BID=3
```

## Build

Enter the shared UVM/VCS environment:

```console
nix develop ../../../../verify
```

Build the DPI reference library:

```console
cargo build --manifest-path casegen/Cargo.toml
```

Compile from this directory:

```console
vcs -full64 -sverilog -timescale=1ns/1ps \
  $VCS_UVM_ARGS \
  -sv_lib casegen/target/debug/libtoint8_casegen \
  -f filelists/toint8_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=toint8_scale_rows_test +BID=3
```

Or via bbdev:

```console
bbdev uvm --build '--ball=toint8' \
  --config sims.verilator.BuckyballPebbleVerilatorConfig \
  --core-config examples/cores/pebble/configs/default.toml
bbdev uvm --run '--ball=toint8 --plusargs +BID=3'
```

## Acceptance (2026-08-04)

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen reuses toint8 model; whole-case DPI API with injected bid
- [x] directed online INT8 scoreboard pass
- [x] signed, zero, rounding, and row-traversal directed cases pass
- [x] protocol + semantic cover enforced (`check_phase` fatal)
- [x] pebble: `+BID=3` config `sims.verilator.BuckyballPebbleVerilatorConfig`
- [ ] toy full: `+BID=5` config `sims.verilator.BuckyballToyVerilatorConfig`
