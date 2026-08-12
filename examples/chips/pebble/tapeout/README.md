# Pebble tapeout contract

The files in this directory are owned by Pebble. `bbdev dc` resolves them from
the Scala config (`sims.verilator.BuckyballPebbleVerilatorConfig`) and passes
the generated DC source list, mapped netlist paths, workload, and power window
through one generated run manifest.

Pebble's main clock is 100 MHz (`10 ns`). `power_sim.sh` produces a time-based
VCD from the gate-level multi-core harness; it must emit `ACTIVITY_FILE`. The
start/end window is in nanoseconds; Pebble
defaults to a 1 ms boot skip and a 4 ms measurement window, without baking a
workload-specific cycle count into bbdev. PrimeTime applies the same window
when it reads VCD/FSDB activity, so startup activity cannot enter the result.
