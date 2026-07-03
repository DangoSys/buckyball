# Fp2IntBall UVM Example

This example verifies the generated `Fp2IntBall` module directly.

## Structure

- `../../../../verify/uvm/src/bb_uvm_pkg.sv`: common Blink UVM transaction items
- `src/if/fp2int_if.sv`: DUT interface
- `src/common/fp2int_defs.svh`: constants and Fp2Int DPI imports
- `src/common/fp2int_items.svh`: Fp2Int-specific transaction extensions
- `src/seq/fp2int_sequences.svh`: directed `fp2int_basic_seq`
- `src/agents/cmd/fp2int_cmd_driver.svh`: drives `cmdReq`
- `src/agents/cmd/fp2int_cmd_monitor.svh`: observes accepted `cmdReq`
- `src/agents/mem/fp2int_mem_model.svh`: responds to bank read/write handshakes
- `src/agents/mem/fp2int_mem_monitor.svh`: observes bank read/write requests
- `src/agents/resp/fp2int_resp_monitor.svh`: observes `cmdResp`
- `src/env/fp2int_scoreboard.svh`: compares observed DUT traffic with the Rust DPI reference
- `src/env/fp2int_env.svh`: wires agents, monitors, memory model, and scoreboard
- `src/tests/fp2int_test.svh`: reset, sequence start, timeout, and objection control
- `src/pkg/fp2int_pkg.sv`: package include entry

The driver publishes the generated stimulus to the memory model and scoreboard.
The scoreboard uses monitored DUT traffic for checks, so the driver is not treated
as proof that the command was accepted by the DUT.

The current directed case covers:

- `INT32` layout: `op1_col = 1`, `wr_col = 1`
- `scale = 1.0` (`0x3F800000`)
- 4 input bank words
- read request checks
- write request checks
- command response checks

The UVM DPI reference and case generator live under `casegen/`:

- `casegen/src/lib.rs` exports DPI-C functions for UVM
- `casegen/src/casegen.rs` generates deterministic UVM cases
- `../emu/src/model.rs` is reused as the fp2int reference model

The smoke sequence asks Rust for case index 0, which is the fixed INT32 smoke
case. Non-zero case indices are seed-based deterministic random cases.

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
  -f filelists/fp2int_ball_toy.f
```

Run:

```console
./simv +UVM_TESTNAME=fp2int_ball_test
```

The same flow can be launched through bbdev:

```console
bbdev uvm --build '--ball=fp2int'
bbdev uvm --run '--ball=fp2int'
```
