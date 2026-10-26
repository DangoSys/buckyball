# VCS workflow

VCS follows the same user-facing stages as Verilator while keeping its own
simulator products below `examples/designs/<design>/output/sims/<entry>/vcs/`:

`vcs.verilog` is implemented under `steps/mill/02_vcs_verilog_event.step.py`
(same mill elaboration as Verilator). Flow:

```text
vcs.verilog → vcs.build → vcs.sim
```

`bbdev vcs --run` routes through all three stages.  The generated RTL is the
standard `BBSimHarness` elaboration; `vcs.build` invokes the independent node under `bebop/src/nodes/vcs`,
which owns the SystemVerilog testbench, DPI shim, build and execution.  The shim supplies BBSimDRAM, SCU UART, and
SCU exit callbacks without linking the Verilator-specific BBSim C++ main.

The VCS flow is intended for RTL functional simulation.  Gate-level power
simulation remains design-owned under `examples/designs/<design>/sims/<entry>/tapeout/`, because
that flow must define the design's workload, reset/boot skip interval, and SDF
policy.
