# TransposeBall UVM Verification

Verifies the generated `TransposeBall` module directly using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/ball/bb_uvm_pkg.sv`: common Blink UVM transaction items and base env
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
- case 2: directed i8 16x32, rows span two groups (`transpose_i8_16x32_test.c`)
- cases 3..22: random with fixed seed `0xCAFE_BABE`, elem_bits in {8,32}, iter in {1,2,4,8,16},
  col in {1,2}, op1/wr group base in 0..3

Each case: reset, preload src words into `mem_model`, drive one command, wait for
`scb.done()` or timeout at 8000 cycles. Scoreboard compares observed writes
(data/addr/group/mask/bank/rob) with DPI `transpose_case_dst_word_*`. `transpose_cov`
requires both `elem_bits` bins, both `col` bins and all `iter` bins hit across the whole test, else
`check_phase` fatal.

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
  -sv_lib ../output/casegen/debug/libtranspose_casegen \
  -f filelists/transpose_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=transpose_ball_test
```

Or via bbdev:

```console
bbdev uvm --run '--design pebble --ball transpose'
```


## Acceptance (2026-08-03)

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen reuses transpose model; whole-case DPI API
- [x] directed i8/i32 scoreboard pass
- [x] deterministic random idx 3..22 pass
- [x] protocol + semantic cover enforced (`check_phase` fatal)
- [x] pebble: 23/23 on the BLink command interface (2026-10-26)
