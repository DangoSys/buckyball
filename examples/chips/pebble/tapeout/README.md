# Pebble tapeout contract

Layout:
- `config.toml` — chip contract (scripts, libs, power window)
- `area/` — DC (`dc.tcl`, `constraints.sdc`)
- `power/` — PTPX + gate sim (`power.tcl`, `power_sim.sh`)
- `ip/` — SRAM MDF

`bbdev dc` resolves paths from `config.toml` and injects `RUN_*`. Timing truth is
`area/constraints.sdc`. Pebble main clock is 100 MHz (`10 ns`).
