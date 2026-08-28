# Im2colBall UVM Verification

Verifies the generated `Im2colBall` module directly using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM transaction items and base env
- `src/common/im2col_defs.svh`: DPI imports and constants
- `src/common/im2col_items.svh`: `im2col_cmd_item` with `load_rust_case`
- `src/seq/im2col_sequences.svh`: `im2col_basic_seq` driving one case per run
- `src/cov/im2col_cov.svh`: ksize/stride/pad/iter coverage
- `src/env/im2col_scoreboard.svh`: preloads mem model, compares writes vs DPI dst words
- `src/env/im2col_env.svh`: extends `bb_blink_env#(1,1)` with scoreboard and cov
- `src/tests/im2col_test.svh`: directed cases 0,1 then random 2..21 with fixed seed
- `src/pkg/im2col_pkg.sv`: package include entry
- `src/tb_top.sv`: instantiates `bb_blink_if#(1,1)` and `Im2colBall`
- `filelists/im2col_ball.f`: VCS/Verilator filelist

## Test plan

`+UVM_TESTNAME=im2col_ball_test` runs:

- case 0: directed k3 6x6 pad0 (ctest `im2col_k3_test.c` data)
- case 1: directed k3 6x6 pad1 (ctest `im2col_k3_pad_test.c` data)
- cases 2..21: random with fixed seed `0xCAFE_BABE`, iter in {3,4,5,6,7},
  ksize in {1,3,5}, stride=1, pad in {0,1}, distinct banks 0..7

Each case: reset, preload src words into `mem_model`, drive one command, wait for
`scb.done()` or timeout at 100000 cycles. Scoreboard compares observed writes
(data/addr/mask/bank/rob) with DPI `im2col_case_dst_word_*`. `im2col_cov`
requires ksize/stride/pad/iter bins hit across the whole test, else
`check_phase` fatal.

Blink command fields (not ISA rs packing):

- `iter` = cmd.iter
- `ksize` = cmd.special[7:0]
- `stride` = cmd.special[15:8]
- `padding` = cmd.special[23:16]
- banks = op1_bank / wr_bank; funct7 is injected from the selected Core ballISA
- TB: op1_col=1, wr_col=1, rs1=0, rs2=0

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. There is no default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=2` |
| `sims.verilator.BuckyballToyVerilatorConfig` | `+BID=3` |

```console
./simv +UVM_TESTNAME=im2col_ball_test +BID=2
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

Compile from this directory (resolve `@UVM@` / `@RTL@` like bbdev):

```console
# Do not pass -hsopt=off: VCS W-2024.09 segfaults on Im2colBall+UVM with it.
vcs -full64 -sverilog -timescale=1ns/1ps \
  $VCS_UVM_ARGS \
  -sv_lib casegen/target/debug/libim2col_casegen \
  -f build/<config>/im2col_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=im2col_ball_test +BID=2
```

Or via bbdev:

```console
bbdev uvm --build '--ball=im2col --config sims.verilator.BuckyballPebbleVerilatorConfig'
bbdev uvm --run '--ball=im2col' -- '+BID=2'
```

## Acceptance (2026-08-04)

- [x] Layout matches Blink common + ball-local split; filelist uses `@UVM@` / `@RTL@`
- [x] casegen pure im2col model (no ExecContext); whole-case DPI API
- [x] directed k3 pad0/pad1 scoreboard pass
- [x] deterministic random idx 2..21 pass
- [x] protocol + semantic cover enforced (`check_phase` fatal)
- [x] pebble: `+BID=2` — 22/22
- [x] toy: `+BID=3` — 22/22
- [x] Note: omit `-hsopt=off` for Im2col (VCS segfault); bbdev skips it for this ball
