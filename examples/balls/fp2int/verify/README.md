# Fp2IntBall UVM Verification

Verifies the generated `Fp2IntBall` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/fp2int_defs.svh`: DPI imports, `fp2int_require_bid`, timeouts
- `src/common/fp2int_items.svh`: `fp2int_cmd_item` with `load_rust_case` (scale in `special[31:0]`)
- `src/seq/fp2int_sequences.svh`: one-case sequence
- `src/cov/fp2int_cov.svh`: layout {i32,i8} and iter {1,4} coverage
- `src/env/fp2int_scoreboard.svh`: preload mem model (group-aware for i8), compare writes vs DPI ref
- `src/env/fp2int_env.svh`: extends `bb_blink_env#(1,1)`
- `src/tests/fp2int_test.svh`: directed 0..1, random 2..11
- `src/pkg/fp2int_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(1,1)` + `Fp2IntBall`
- `filelists/fp2int_ball.f`: VCS filelist

## Test plan

`+UVM_TESTNAME=fp2int_ball_test` runs:

- case 0: directed INT32 (`op1_col=1`, `wr_col=1`, scale=1.0)
- case 1: directed INT8 (`op1_col=4`, `wr_col=1`, scale=2.0)
- cases 2..11: random INT32 with fixed seed `0xBEEF_0001`

Each case: reset, preload src words into `mem_model` (INT8 uses `group_id` 0..3),
drive one command, wait for `scb.done()` or timeout at 400 cycles. Scoreboard
compares observed writes with DPI `fp2int_ref_i32` / `fp2int_ref_i8`. `fp2int_cov`
requires layout and iter bins hit across the whole test, else `check_phase` fatal.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=3` |
| `sims.verilator.BuckyballToyVerilatorConfig` (full.toml) | `+BID=5` |

```console
./simv +UVM_TESTNAME=fp2int_ball_test +BID=3
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
./simv +UVM_TESTNAME=fp2int_ball_test +BID=3
```

Or via bbdev:

```console
bbdev uvm --build '--ball=fp2int'
bbdev uvm --run '--ball=fp2int' -- '+BID=3'
```

## Acceptance (2026-08-04)

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen reuses fp2int model; whole-case DPI API with injected bid
- [x] directed INT32/INT8 scoreboard pass
- [x] deterministic random idx 2..11 pass
- [x] protocol + semantic cover enforced (`check_phase` fatal)
- [x] pebble: `+BID=3` config `sims.verilator.BuckyballPebbleVerilatorConfig` — 12/12
- [x] toy full: `+BID=5` config `sims.verilator.BuckyballToyVerilatorConfig` — 12/12
