# TransposeBall UVM Verification

Verifies the generated `TransposeBall` module directly using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM transaction items and base env
- `src/common/transpose_defs.svh`: DPI imports and constants
- `src/common/transpose_items.svh`: `transpose_cmd_item` with `load_rust_case`
- `src/seq/transpose_sequences.svh`: `transpose_basic_seq` driving one case per run
- `src/cov/transpose_cov.svh`: `elem_bits` {8,32} and `iter` {1,2,4,8,16} coverage
- `src/env/transpose_scoreboard.svh`: preloads mem model, compares writes vs DPI dst words
- `src/env/transpose_env.svh`: extends `bb_blink_env#(1,1)` with scoreboard and cov
- `src/tests/transpose_test.svh`: directed cases 0,1 then random 2..21 with fixed seed
- `src/pkg/transpose_pkg.sv`: package include entry
- `src/tb_top.sv`: instantiates `bb_blink_if#(1,1)` and `TransposeBall`
- `filelists/transpose_ball.f`: VCS/Verilator filelist

## Test plan

`+UVM_TESTNAME=transpose_ball_test` runs:

- case 0: directed i8, iter=16
- case 1: directed i32, iter=8
- cases 2..21: random with fixed seed `0xCAFE_BABE`, elem_bits in {8,32}, iter in {1,2,4,8,16}

Each case: reset, preload src words into `mem_model`, drive one command, wait for
`scb.done()` or timeout at 2000 cycles. Scoreboard compares observed writes
(data/addr/mask/bank/rob) with DPI `transpose_case_dst_word_*`. `transpose_cov`
requires both `elem_bits` bins and all `iter` bins hit across the whole test, else
`check_phase` fatal.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. There is no default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballToyVerilatorConfig` | `+BID=2` |
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=0` |

```console
./simv +UVM_TESTNAME=transpose_ball_test +BID=2
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
  -sv_lib casegen/target/debug/libtranspose_casegen \
  -f filelists/transpose_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=transpose_ball_test +BID=2
```

Or via bbdev:

```console
bbdev uvm --build '--ball=transpose'
bbdev uvm --run '--ball=transpose' -- '+BID=2'
```


## Acceptance (2026-08-03)

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen reuses transpose model; whole-case DPI API
- [x] directed i8/i32 scoreboard pass
- [x] deterministic random idx 2..21 pass
- [x] protocol + semantic cover enforced (`check_phase` fatal)
- [x] toy: `+BID=2` config `sims.verilator.BuckyballToyVerilatorConfig` — 22/22
- [x] pebble: `+BID=0` config `sims.verilator.BuckyballPebbleVerilatorConfig` — 22/22
