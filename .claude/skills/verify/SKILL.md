---
name: verify
description: Verify functional correctness of the Ball named $ARGUMENTS. Use this skill when users ask to verify/test a Ball, check whether a Ball works correctly, or validate a newly created Ball.
---

**Important: build, simulation, and test operations must be invoked via MCP tools. Do not call bbdev CLI or nix develop directly.**

## Phase 1 - Completeness Check

Use `/check` logic to validate registration consistency, then ensure all required artifacts exist and fill missing pieces:
1. Ball implementation: `arch/src/main/scala/framework/balldomain/prototype/<name>/`
2. Registration entry in `arch/src/main/scala/framework/balldomain/configs/default.json`
3. ISA macro file in `bb-tests/workloads/lib/bbhw/isa/`
4. CTest file in `bb-tests/workloads/src/CTest/toy/`
5. Sardine workload entry in `bb-tests/sardine/tests/test_ctest.py` (`ctest_workloads`)

## Phase 2 - Build and Simulate

1. Run MCP tool `bbdev_workload_build` to build all CTests
2. Run MCP tool `bbdev_verilator_run` for this Ball's CTest
   - binary naming format: `ctest_<name>_test_singlecore-baremetal`
   - set `batch=true`
3. If build/simulation fails, switch to `/debug` flow

## Phase 3 - Coverage Analysis

1. Run MCP tool `bbdev_sardine_run` with `coverage=true`
2. Read coverage reports:
   - line coverage data under `bb-tests/sardine/reports/coverage/annotated/`
   - locate target Ball file by searching Ball class name
3. Analyze RTL line coverage of the Ball:
   - inspect uncovered lines (marked `000000`)
   - focus on FSM states, boundary conditions, and error paths
4. If coverage is insufficient, propose or add tests, then rebuild and rerun simulation

## Phase 4 - PMC Performance Analysis

After simulation passes, analyze PMC traces from `bdb.log`:

1. Locate log directory (`ls -t arch/log/ | head -5`)
2. Search `[PMCTRACE] BALL` entries in `bdb.log` and extract elapsed cycles for the target Ball
3. Summarize:
   - average elapsed cycles per task
   - max/min elapsed cycles
   - total invocation count

## Phase 5 - Waveform Analysis (when simulation fails)

If simulation fails, use waveform-mcp for precise timing analysis in addition to logs. See `/waveform`.

Key signal checklist:
- `cmdReq.valid && cmdReq.ready` (command handshake)
- SRAM `req.valid/ready` and `resp.valid` (read/write timing)
- FSM state register (state transitions)
- `cmdResp.valid && cmdResp.fire` (completion handshake)

## Failure Handling

If simulation result is FAILED, run `/debug` for systematic troubleshooting.
