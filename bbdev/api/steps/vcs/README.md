# VCS workflow

VCS follows the same user-facing stages as Verilator while keeping its own
simulator products below `arch/build/<config>/vcs/`:

```text
vcs.verilog → vcs.build → vcs.sim
```

`bbdev vcs --run` routes through all three stages.  The generated RTL is the
standard `BBSimHarness` elaboration; `vcs.build` adds a small SystemVerilog
testbench and a VCS-only DPI shim.  The shim supplies BBSimDRAM, SCU UART, and
SCU exit callbacks without linking the Verilator-specific BBSim C++ main.

The VCS flow is intended for RTL functional simulation.  Gate-level power
simulation remains chip-owned under `examples/chips/<chip>/tapeout/`, because
that flow must define the chip's workload, reset/boot skip interval, and SDF
policy.
