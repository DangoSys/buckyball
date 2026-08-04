# Int2FpBall UVM Verification

Verifies the generated `Int2FpBall` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/int2fp_defs.svh`: DPI imports, `int2fp_require_bid`, timeouts
- `src/common/int2fp_items.svh`: `int2fp_cmd_item` with `load_rust_case`
  (`special[31:0]=scale`, `special[33:32]=output_mode`)
- `src/seq/int2fp_sequences.svh`: one-case sequence
- `src/cov/int2fp_cov.svh`: mode {fp32,i8} and iter {1,2,4,16} coverage
- `src/env/int2fp_scoreboard.svh`: preload mem model (group-aware for INT8), compare writes vs DPI
- `src/env/int2fp_env.svh`: extends `bb_blink_env#(1,1)`
- `src/tests/int2fp_test.svh`: directed 0..1, random 2..21 with seed `0xCAFE_BABE`
- `src/pkg/int2fp_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(1,1)` + `Int2FpBall`
- `filelists/int2fp_ball.f`: VCS filelist (`@UVM@` / `@RTL@`)

## Test plan

`+UVM_TESTNAME=int2fp_ball_test` runs:

- case 0: directed INT32→FP32 (`op1_col=1`, `wr_col=1`, `output_mode=0`, scale=1.0)
- case 1: directed INT32→INT8 (`op1_col=4`, `wr_col=1`, `output_mode=1`, scale=0.5)
- cases 2..21: random FP32/INT8 with fixed seed `0xCAFE_BABE`

Each case: reset, preload src words into `mem_model` (INT8 uses `group_id` 0..3),
drive one command, wait for `scb.done()` or timeout at 4000 cycles. Scoreboard
compares observed writes with DPI `int2fp_ref_fp32` / `int2fp_ref_i8`. `int2fp_cov`
requires mode and iter bins hit across the whole test, else `check_phase` fatal.

`funct7` is 52 (`BB_INT2FP_FUNC7`).

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=4` |
| `sims.verilator.BuckyballToyVerilatorConfig` (full.toml) | `+BID=6` |

```console
./simv +UVM_TESTNAME=int2fp_ball_test +BID=4
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

Compile from this directory (resolve `@UVM@` / `@RTL@` first, or use bbdev):

```console
vcs -full64 -sverilog -timescale=1ns/1ps \
  $VCS_UVM_ARGS \
  -sv_lib casegen/target/debug/libint2fp_casegen \
  -f filelists/int2fp_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=int2fp_ball_test +BID=4
```

Or via bbdev:

```console
bbdev uvm --build '--ball=int2fp' --config sims.verilator.BuckyballPebbleVerilatorConfig
bbdev uvm --run '--ball=int2fp' --config sims.verilator.BuckyballPebbleVerilatorConfig -- '+BID=4'
```
