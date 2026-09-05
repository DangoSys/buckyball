# Int2FpBall UVM Verification

Verifies the generated `Int2FpBall` module using the shared Blink UVM framework.

RTL contract: one mnemonic `INT32_TO_FP32`, `inBW=2` (input + scale banks),
`outBW=1`, `rs2[0]=relu`, `rs2[63:1]=0`, iter a positive multiple of 4.
Scales are bank rows, not MMIO.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM agents and base env
- `src/common/int2fp_defs.svh`: DPI imports, `INT32_TO_FP32_FUNCT7`, timeouts
- `src/common/int2fp_items.svh`: `int2fp_cmd_item` with `load_rust_case`
- `src/seq/int2fp_sequences.svh`: one-case sequence
- `src/cov/int2fp_cov.svh`: relu `{0,1}` and iter `{4,8,16}`, 100% or fatal
- `src/env/int2fp_scoreboard.svh`: preload src/scale banks, compare writes vs DPI dst
- `src/env/int2fp_env.svh`: extends `bb_blink_env #(BB_IN_BW, BB_OUT_BW)`
- `src/tests/int2fp_test.svh`: `int2fp_ball_test` loops cases 0..5
- `src/pkg/int2fp_pkg.sv`: package entry
- `src/tb_top.sv`: `bb_blink_if #(BB_IN_BW, BB_OUT_BW)` + `Int2FpBall` (2 read, 1 write)
- `filelists/int2fp_ball.f`: VCS filelist (`@UVM@` / `@RTL@`)

## Test plan

`+UVM_TESTNAME=int2fp_ball_test` runs six directed cases:

| Index | iter | relu | Intent |
|-------|------|------|--------|
| 0 | 4 | 0 | MLIR-scale values |
| 1 | 4 | 1 | relu clamps negatives |
| 2 | 8 | 0 | eight-row, no relu |
| 3 | 8 | 1 | eight-row, relu |
| 4 | 16 | 0 | sixteen-row, no relu |
| 5 | 16 | 1 | sixteen-row, relu |

Each case: reset, preload 16-byte i32 input rows and f32 scale rows into
`mem_model`, drive one `INT32_TO_FP32` command, wait for `scb.done()` or
timeout at 20000 cycles. Expected dst = `f32(i32) * scale` with relu clamp
on the integer before convert. `int2fp_cov` fatals unless relu and iter bins
are 100%.

## BID (required)

`+BID=<n>` is mandatory. Missing it fatals at test start. No default bid.

| Config | Plusarg |
|--------|---------|
| `sims.verilator.BuckyballPebbleVerilatorConfig` | `+BID=4` |

```console
./simv +UVM_TESTNAME=int2fp_ball_test +BID=4
```

## Build / Run

Enter the shared UVM/VCS environment and build the DPI reference library:

```console
nix develop ../../../../verify
cargo build --manifest-path casegen/Cargo.toml
```

Human:

```console
bbdev uvm --run '--chip pebble'
bbdev uvm --run '--chip pebble --ball int2fp'
```

Agent (MCP):

```text
bbdev_uvm_run(chip="pebble", ball="int2fp")
```

bbdev injects `INT32_TO_FP32_FUNCT7`, `BB_IN_BW`, `BB_OUT_BW`, `+BID`, and
`+UVM_TESTNAME=int2fp_ball_test` from pebble chip.pb.
