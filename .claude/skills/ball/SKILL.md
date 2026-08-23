---
name: ball
description: Create a new Buckyball Ball operator named $ARGUMENTS, covering the full flow from implementation to verification.
---

**Important: all build/simulation operations must go through MCP tools from project `.mcp.json` (`validate`, `bbdev_workload_build`, `bbdev_bemu_sim`, `bbdev_bebop_verilator_run`, etc.). Do not use bbdev CLI or nix develop directly. If `buckyball-dev` is not loaded, stop and report it.**

## Phase 1 - Requirement Collection

1. Inspect registration state and decide `ballId` + `funct7`:
   - active file from `examples/chips/<chip>/configs/tiles/cores/default.toml` → `balldomain = ...`
   - usually `examples/chips/<chip>/configs/tiles/cores/balldomains/*.toml`
2. Check for partial existing implementation (incremental mode):
   - existing directory in `examples/balls/`
   - existing ISA macro in `examples/balls/<name>/workloads/isa/` (or base under `bb-tests/workloads/lib/bbhw/isa/`)
   - existing chip/ball ctests under `examples/` or `bb-tests/workloads/`
3. Confirm with user:
   - target chip
   - operator semantics
   - `inBW` / `outBW`
   - whether `op2` is needed
   - meaning of `iter`

## Phase 2 - Implement the Ball

1. Read references:
   - simple example: `.../prototype/relu/ReluBall.scala`, `Relu.scala`
   - complex example: `.../prototype/systolicarray/`
   - Blink protocol: `.../blink/blink.scala`, `bank.scala`, `status.scala`
   - SRAM IO: `.../memdomain/backend/banks/SramIO.scala`
2. Create files under `examples/balls/<name>/arch/src/main/scala/` using templates from `references/`.

### Key constraints
- SRAM read latency is 1 cycle (`resp.valid` in the cycle after `req.fire`)
- Latch command fields when `cmdReq.fire`
- Base FSM pattern: `idle -> read -> compute -> write -> complete -> idle`
- `status.idle` and `status.running` must map correctly to FSM states

## Phase 3 - Register the Ball

Edit the chip balldomain TOML (the one selected by `cores/default.toml`, or the variant you are changing):

1. Append a `ballIdMappings` row (`ballId`, `ballName`, `ballClass`, `config`, `inBW`, `outBW`)
2. Update `ballNum`
3. Append a `ballISA` row (`mnemonic`, `funct7`, `bid`)
4. Run MCP `validate(chip=..., balldomain=...)` before continuing

## Phase 4 - Add ISA C Macro

Create `examples/balls/<name>/workloads/isa/<name>.h` (include `<bbhw/isa/isa.h>`),
then `#include <isa/<name>.h>` from the ball's ctests. Do **not** add ball ISA to
central `bb-tests/workloads/lib/bbhw/isa/isa.h` (base mem/frontend only).

## Phase 5 - Add CTest

1. Create `<name>_test.c` under `bb-tests/workloads/src/CTest/toy/`
2. Register in `bb-tests/workloads/src/CTest/toy/CMakeLists.txt` using `add_cross_platform_test_target`

## Phase 6 - Validate, Build, and Simulate

1. Run `validate` and ensure all 6 invariants pass
2. Run `bbdev_workload_build(chip="toy")` (or the target chip)
3. Run `bbdev_bemu_sim` for this Ball's CTest binary (functional first)
4. Run `bbdev_bebop_verilator_run` with an explicit config (e.g. `sims.verilator.BuckyballToyVerilatorConfig`)
5. Interpret results:
   - `PASSED` -> done
   - bemu pass / verilator fail -> RTL/timing issue, switch to `/debug`
   - bemu fail -> fix workload / ball semantics first
