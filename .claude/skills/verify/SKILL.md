---
name: verify
description: Verify functional correctness of the Ball named $ARGUMENTS. Use this skill when users ask to verify/test a Ball, check whether a Ball works correctly, or validate a newly created Ball.
---

**Important: build, simulation, and test operations must be invoked via MCP tools from project `.mcp.json`. Do not call bbdev CLI or nix develop directly. If `buckyball-dev` is not loaded, stop and report it.**

## Phase 1 - Completeness Check

Use `/check` logic to validate registration consistency, then ensure all required artifacts exist and fill missing pieces:
1. Ball implementation: `examples/balls/<name>/arch/src/main/scala/`
2. Registration entry in chip balldomain TOML under `examples/chips/<chip>/configs/tiles/cores/balldomains/`
3. ISA macro file in `bb-tests/workloads/lib/bbhw/isa/`
4. CTest for that ball/chip

## Phase 2 - Build and Simulate

1. Run `bbdev_workload_build(chip=...)` to build CTests
2. Run `bbdev_bemu_sim(chip=..., binary=...)` first
3. Run `bbdev_bebop_verilator_run(binary=..., config=...)` for RTL
   - binary is `{chip}_{stem}-{platform}`, e.g. `toy_transpose_test-singlecore-baremetal`
   - config is required, e.g. `sims.verilator.BuckyballToyVerilatorConfig`
4. If build/simulation fails, switch to `/debug` flow

## Phase 3 - PMC Performance Analysis

After simulation passes, analyze PMC traces from `bdb.log`:

1. Locate log directory (`ls -t log/ | head -5`)
2. Search `[PMCTRACE] BALL` entries in `bdb.log` and extract elapsed cycles for the target Ball
3. Summarize:
   - average elapsed cycles per task
   - max/min elapsed cycles
   - total invocation count

## Phase 4 - Waveform Analysis (when simulation fails)

If simulation fails, use waveform-mcp for precise timing analysis in addition to logs. See `/waveform`.

Key signal checklist:
- `cmdReq.valid && cmdReq.ready` (command handshake)
- SRAM `req.valid/ready` and `resp.valid` (read/write timing)
- FSM state register (state transitions)
- `cmdResp.valid && cmdResp.fire` (completion handshake)

## Failure Handling

If simulation result is FAILED, run `/debug` for systematic troubleshooting.
