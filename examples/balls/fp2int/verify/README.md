# Fp2IntBall UVM Verification

Verifies the generated `Fp2IntBall` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/fp2int_defs.svh`: DPI imports, `fp2int_require_bid`, timeouts
- `src/common/fp2int_items.svh`: `fp2int_cmd_item` with `load_rust_case` (Da address in `special[12:0]`)
- `src/seq/fp2int_sequences.svh`: one-case sequence
- `src/cov/fp2int_cov.svh`: online INT8 layout and iter 1 coverage
- `src/env/fp2int_scoreboard.svh`: preload mem model (group-aware for i8), compare writes vs DPI ref
- `src/env/fp2int_env.svh`: extends `bb_blink_env#(1,1)`
- `src/tests/fp2int_*_test.svh`: one directed case per UVM test
- `src/pkg/fp2int_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(1,1)` + `Fp2IntBall`
- `filelists/fp2int_ball.f`: VCS filelist

## Test plan

The test names are:

- `fp2int_signed_test`: signed values across all four source groups
- `fp2int_zero_test`: all-zero activation (`Da=1.0`)
- `fp2int_rounding_test`: ties-to-even quantization boundaries
- `fp2int_rows_test`: two-row scan and packed output traversal
- `fp2int_scale_rows_test`: maximum value is in the second row

Each case: reset, preload src words into `mem_model` (INT8 uses `group_id` 0..3),
drive one command, wait for `scb.done()` or timeout at 400 cycles. Scoreboard
compares observed writes with DPI `fp2int_ref_i8`. `fp2int_cov` requires the online
INT8 layout and one-row iteration bin, else `check_phase` fatal.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=3` |
| `sims.verilator.BuckyballToyVerilatorConfig` (full.toml) | `+BID=5` |

```console
./simv +UVM_TESTNAME=fp2int_scale_rows_test +BID=3
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
  -sv_lib casegen/target/debug/libfp2int_casegen \
  -f filelists/fp2int_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=fp2int_scale_rows_test +BID=3
```

Or via bbdev:

```console
bbdev uvm --build '--ball=fp2int' \
  --config sims.verilator.BuckyballPebbleVerilatorConfig \
  --core-config examples/cores/pebble/configs/default.toml
bbdev uvm --run '--ball=fp2int --plusargs +BID=3'
```

## Acceptance (2026-08-04)

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen reuses fp2int model; whole-case DPI API with injected bid
- [x] directed online INT8 scoreboard pass
- [x] signed, zero, rounding, and row-traversal directed cases pass
- [x] protocol + semantic cover enforced (`check_phase` fatal)
- [x] pebble: `+BID=3` config `sims.verilator.BuckyballPebbleVerilatorConfig`
- [ ] toy full: `+BID=5` config `sims.verilator.BuckyballToyVerilatorConfig`
