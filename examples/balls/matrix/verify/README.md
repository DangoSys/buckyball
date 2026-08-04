# MatrixBall UVM Verification

Verifies the generated `MatrixBall` module using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/matrix_defs.svh`: DPI imports, `matrix_require_bid`, timeouts
- `src/common/matrix_items.svh`: `matrix_cmd_item` with `load_rust_case`
- `src/seq/matrix_sequences.svh`: one-case sequence
- `src/cov/matrix_cov.svh`: mode {OS,WS} and M/N/K shape bins
- `src/env/matrix_scoreboard.svh`: preload A/B, compare writes vs DPI expected list
- `src/env/matrix_env.svh`: extends `bb_blink_env#(2,4)`
- `src/tests/matrix_test.svh`: directed 0..2, random 3..22
- `src/pkg/matrix_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if#(2,4)` + `MatrixBall` (2 read, 4 write ports)
- `filelists/matrix_ball.f`: VCS filelist

## Test plan

`+UVM_TESTNAME=matrix_ball_test` runs:

- case 0: OS 4x4x4 (banks 0,1,2)
- case 1: WS 4x4x4
- case 2: OS 16x16x16
- cases 3..22: random M/N/K in {1,2,4,8,16}, mode 0/1, distinct banks 0..7

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
./simv +UVM_TESTNAME=matrix_ball_test +BID=4
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
  -f filelists/matrix_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=matrix_ball_test +BID=4
```

Or via bbdev:

```console
bbdev uvm --build '--ball=matrix'
bbdev uvm --run '--ball=matrix' -- '+BID=4'
```

## Known DUT issue

After mem-model multi-port fix (VCS no longer segfaults), pebble `+BID=1`
still hangs on directed OS 4x4: reads complete (`r0=4 r1=4`) but only
`wr=2/4` store rows appear and `cmd_resp` never fires. TIMEOUT fatals with
those counts. Treat as DUT/RTL investigation; UVM stack is intentionally
fail-hard so the hang is visible.

## Acceptance

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen whole-case DPI API; `cargo test` in casegen
- [x] VCS build with `matrix_ball.f` (no segfault on Instantiate DUT)
- [x] `+BID=` required; pebble `+BID=1` / toy `+BID=4`
- [ ] 23/23 functional pass — blocked on DUT hang above (`wr=2/4`)
