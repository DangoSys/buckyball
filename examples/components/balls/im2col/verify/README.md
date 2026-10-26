# Im2colBall UVM Verification

Verifies the generated `Im2colBall` module directly using the shared Blink UVM framework.

## Structure

- `../../../../verify/uvm/src/ball/bb_uvm_pkg.sv`: common Blink UVM transaction items and base env
- `src/common/im2col_defs.svh`: DPI imports and constants
- `src/common/im2col_items.svh`: `im2col_cmd_item` with `load_rust_case`
- `src/seq/im2col_sequences.svh`: `im2col_basic_seq` driving one case per run
- `src/cov/im2col_cov.svh`: kernel/stride/padding/start/base/window/tile_rows/group coverage
- `src/env/im2col_scoreboard.svh`: preloads every input group, checks reads and writes in issue order
- `src/env/im2col_env.svh`: extends `bb_blink_env#(1,1)` with scoreboard and cov
- `src/tests/im2col_test.svh`: runs the directed cases in order
- `src/pkg/im2col_pkg.sv`: package include entry
- `src/tb_top.sv`: instantiates `bb_blink_if#(1,1)` and `Im2colBall`
- `filelists/im2col_ball.f`: VCS/Verilator filelist

## Test plan

`casegen/src/model.rs` is the reference (same semantics as `emu/src/im2col.rs`):
input group g is channel block g, K tile `t = (kr * kernel + kc) * in_groups + g`
copies one input row, output group `t / G` (`G = K_TILES / wr_col`) line
`((w / tile_rows) * G + t % G) * tile_rows + w % tile_rows`. The scoreboard checks
every read (group, line) and every write (group, line, data) in the Ball's issue
order: K tile outer, window inner.

Directed cases (`casegen/src/casegen.rs`):

- 0: 1x1 kernel, 2x2 input, two input and two output groups
- 1: 3x3 kernel, 4x4 input, nine output groups (one tap each), tile_rows 8
- 2: 3x3 kernel, padding 1, 2x2 input, one output group, tile_rows 4
- 3: bottom/right padding (start_row = start_col = 1), 2x2 kernel, four output groups
- 4: window range 5..7 at input_base 2, partial M tile zero-filled
- 5: 5x5 kernel, tile_rows 1, five output groups (one image row each)
- 6: 7x7 stride-2 kernel, 9x9 input, seven output groups
- 7: 2x2 kernel, two input groups, input_base 3, windows 1..2, eight output groups

Op1/wr base groups are nonzero in several cases.

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
  -sv_lib ../output/casegen/debug/libim2col_casegen \
  -f build/<config>/im2col_ball.f
```

Run:

```console
./simv +UVM_TESTNAME=im2col_ball_test +BID=2
```

Or via bbdev:

```console
bbdev uvm --run '--design pebble --ball im2col'
```
