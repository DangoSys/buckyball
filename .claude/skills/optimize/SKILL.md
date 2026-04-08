---
name: optimize
description: Analyze and optimize latency and area for the RTL module named $ARGUMENTS. Applies to any RTL module (Ball, MemFrontend, BBus, GlobalROB, etc.), not limited to Ball operators.
---

**Important: synthesis and simulation operations must be invoked through MCP tools. Do not call bbdev CLI or nix develop directly.**

## Phase 1 - Baseline Measurement

1. Run MCP tool `bbdev_yosys_synth` for Yosys synthesis + OpenSTA timing analysis
2. Read area report: `bbdev/api/steps/yosys/log/hierarchy_report.txt` (submodule breakdown)
3. Read timing report: `bbdev/api/steps/yosys/log/timing_report.txt` (critical paths)

### How to parse `hierarchy_report.txt`

This report uses Yosys `stat -top` format. Key fields:
- `Chip area for module` followed by submodule area breakdown
- `Number of cells` - cell count
- `Sequential` - register area (flip-flops)
- `Combinational` - combinational logic area
- search target module name to locate hierarchy

### How to parse `timing_report.txt`

OpenSTA output format:
- `Startpoint:` / `Endpoint:` - critical path endpoints
- `Path Delay` - path delay
- `Slack` - timing margin (negative means violation)
- search target module name to see whether it is on critical paths

### Latency measurement

**For Ball modules:**
1. Extract elapsed cycles from PMC traces in `bdb.log` (`[PMCTRACE] BALL ball_id=X`)
2. If CTest uses `read_rdcycle()`, also collect cycle counts from `stdout.log`
3. If neither is available, measure precisely with waveform-mcp:
   - find `cmdReq.valid && cmdReq.ready` timestamps via `find_conditional_events`
   - find `cmdResp.valid` timestamps
   - difference is operation latency

**For non-Ball modules:**
1. Use waveform-mcp to measure cycle counts of key operations
2. Or add `read_rdcycle()` instrumentation in CTests

## Phase 2 - Area Analysis

Extract area data for target module and submodules from `hierarchy_report.txt`:
- total area (`Chip area`)
- sequential ratio (register area)
- combinational ratio (logic area)
- submodule area ranking

Identify major area contributors:
- high sequential ratio -> many registers; consider SRAM replacement
- high combinational ratio -> complex logic; consider simplification/sharing

Compare with similar modules to identify efficiency gaps.

## Phase 3 - Timing/Latency Analysis

1. Check whether critical paths pass through target module and inspect path delays in `timing_report.txt`
2. Measure operation latency from PMC traces or waveform data
3. For Ball/FSM modules, analyze FSM in Chisel source:
   - draw FSM state transitions
   - estimate cycles per state (best/worst case)
   - identify bottleneck states
   - analyze SRAM access pattern (serial vs pipelined vs multi-port parallel)

## Phase 4 - Optimization Proposals

Provide quantifiable optimization options. Each option should include:
- optimization method
- expected area delta (quantified from hierarchy report)
- expected latency delta (cycles)
- expected frequency impact
- trade-off explanation

Common optimization patterns:

**Reduce latency:**
- read/write pipelining: overlap writes with next-round reads; slight area increase, notable latency drop
- multi-bank parallel read: use `inBW > 1` to read multiple rows concurrently; area stable, latency drops with port count
- remove idle wait states: merge unnecessary FSM states; slight area reduction
- compute while reading: overlap computation with SRAM response timing

**Reduce area:**
- replace large reg arrays with SRAM accesses; strong area reduction, possible latency increase
- share compute units across operations (time-multiplexing); lower area, possible latency increase
- reduce counter bit-width based on real bounds; small area reduction

**Improve frequency:**
- split long combinational paths by adding pipeline regs; slight area increase, possibly +1 cycle latency

Present options and let the user choose.

## Phase 5 - Implementation

Apply the selected optimization in Chisel code.

## Phase 6 - Post-Optimization Measurement

1. Run `bbdev_yosys_synth` again and compare hierarchy reports
2. If CTest exists, rerun simulation with `bbdev_verilator_run` and compare cycle counts in PMC traces
3. Run `validate` to ensure registration consistency is preserved
4. Output before/after comparison report:

| Metric | Before | After | Delta |
|------|--------|--------|------|
| Area | X | Y | -Z% |
| Cycles | A | B | -C% |
| Critical path delay | D | E | -F% |

## Troubleshooting

If MCP tools return HTTP 500 or `returncode=1`:
- read `bbdev/server.log` for detailed error information
