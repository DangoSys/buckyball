# SMatMulBall UVM Verification

Verifies the generated `SMatMulBall` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/smatmul_defs.svh`: DPI imports, `matrix_require_bid`, timeouts
- `src/common/smatmul_items.svh`: `smatmul_cmd_item` with `load_rust_case`
- `src/seq/smatmul_sequences.svh`: one-case sequence
- `src/cov/smatmul_cov.svh`: M/N/K shape bins `{1,2,4,8,16}`
- `src/env/smatmul_scoreboard.svh`: preload A/B, compare writes vs DPI expected list
- `src/env/smatmul_env.svh`: extends `bb_blink_env#(2,4)`
- `src/tests/smatmul_test.svh`: directed 0..3, random 4..23
- `src/pkg/smatmul_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(2,2)` + `SMatMulBall` (2 read, 2 write ports)
- `filelists/smatmul_ball.f`: VCS filelist

## Test plan

`+UVM_TESTNAME=smatmul_ball_test` runs:

- case 0: 4x4x4 OS (banks 0,1,2)
- case 1: 5x7x3 OS tail
- case 2: 16x16x16 OS
- case 3: 32x16x16 WS
- cases 4..23: random M in {1,2,4,8,16,32}, N/K in {1,2,4,8,16}, distinct banks 0..7

Each case: reset, preload A/B into mem model, drive one command, wait for
`scb.done()` or timeout at 100000 cycles. Scoreboard compares writes
(group/addr/data/mask) in DPI emission order.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=1` |
| `sims.verilator.BuckyballToyVerilatorConfig` | `+BID=4` |

```console
./simv +UVM_TESTNAME=smatmul_ball_test +BID=4
```

## Build

```console
nix develop ../../../../verify
cargo build --manifest-path casegen/Cargo.toml
```

Compile from this directory:

```console
vcs -full64 -sverilog -timescale=1ns/1ps \
  $VCS_UVM_ARGS \
  -sv_lib casegen/target/debug/libmatrix_casegen \
  -f filelists/smatmul_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=smatmul_ball_test +BID=4
```

Or via bbdev:

```console
bbdev uvm --build '--ball=matrix'
bbdev uvm --run '--ball=matrix' -- '+BID=4'
```

## Acceptance

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen whole-case DPI API; `cargo test` in casegen
- [x] VCS build with `smatmul_ball.f` (no segfault on Instantiate DUT)
- [x] `+BID=` required; pebble `+BID=1` / toy `+BID=4`
- [ ] 23/23 functional pass
