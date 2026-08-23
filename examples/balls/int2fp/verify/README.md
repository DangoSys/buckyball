# Int2FpBall UVM Verification

Verifies the generated `Int2FpBall` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/int2fp_defs.svh`: DPI imports, `int2fp_require_bid`, timeouts
- `src/common/int2fp_items.svh`: `int2fp_cmd_item` with `load_rust_case`
  (`special[12:0]=Da`, `special[25:13]=Dw`; tensor/channel use separate funct7)
- `src/seq/int2fp_sequences.svh`: one-case sequence
- `src/cov/int2fp_cov.svh`: iter {1,16} coverage
- `src/env/int2fp_scoreboard.svh`: preload mem model and compare writes vs DPI
- `src/env/int2fp_env.svh`: extends `bb_blink_env#(1,1)`
- `src/tests/int2fp_*_test.svh`: one directed case per UVM test
- `src/pkg/int2fp_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(1,1)` + `Int2FpBall`
- `filelists/int2fp_ball.f`: VCS filelist (`@UVM@` / `@RTL@`)

## Test plan

The test names are:

- `int2fp_tensor_rows_test`: 16-row tensor scale
- `int2fp_channel_lanes_test`: four-group channel scale with distinct `Dw`
- `int2fp_tensor_groups_test`: four-group tensor scale traversal
- `int2fp_channel_base_test`: channel scale at MMIO byte address 64
- `int2fp_channel_two_rows_test`: two rows reuse all sixteen channel scales

Each case: reset, preload src words and the four Da/Dw MMIO bytes into `mem_model`,
drive one command, wait for `scb.done()` or timeout at 4000 cycles. Scoreboard
compares observed writes with DPI `int2fp_ref_fp32`, including `Da * Dw` for
per-tensor and per-channel weights. The tensor path requires eight MMIO byte
reads (four Da bytes followed by four tensor-Dw bytes).

The core's `ballISA` assigns separate funct7 values to `INT2FP_TENSOR` and
`INT2FP_CHANNEL` (Pebble uses 52 and 54). The Ball does not define these
numbers.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=4` |
| `sims.verilator.BuckyballToyVerilatorConfig` (full.toml) | `+BID=6` |

```console
./simv +UVM_TESTNAME=int2fp_channel_two_rows_test +BID=4
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
./simv +UVM_TESTNAME=int2fp_channel_two_rows_test +BID=4
```

Or via bbdev:

```console
bbdev uvm --build '--ball=int2fp' --config sims.verilator.BuckyballPebbleVerilatorConfig \
  --core-config examples/cores/pebble/configs/default.toml
bbdev uvm --run '--ball=int2fp --test int2fp_channel_two_rows_test --plusargs +BID=4'
```
