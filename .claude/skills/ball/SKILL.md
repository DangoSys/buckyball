---
name: ball
description: Create a new Buckyball Ball operator named $ARGUMENTS, covering the full flow from implementation to verification.
---

**Important: all build/simulation operations must go through MCP tools (`validate`, `bbdev_workload_build`, `bbdev_verilator_run`, etc.). Do not use bbdev CLI or nix develop directly.**

## Phase 1 - Requirement Collection

1. Inspect registration state and decide `ballId` + `funct7`:
   - `arch/src/main/scala/framework/balldomain/configs/default.json`
   - `arch/src/main/scala/examples/toy/balldomain/DISA.scala`
2. Check for partial existing implementation (incremental mode):
   - existing directory in `arch/src/main/scala/framework/balldomain/prototype/`
   - existing ISA macro in `bb-tests/workloads/lib/bbhw/isa/`
   - existing CTest in `bb-tests/workloads/src/CTest/toy/`
3. Confirm with user:
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
2. Create files under `arch/src/main/scala/framework/balldomain/prototype/<name>/` using templates from `references/`.

### Key constraints
- SRAM read latency is 1 cycle (`resp.valid` in the cycle after `req.fire`)
- Latch command fields when `cmdReq.fire`
- Base FSM pattern: `idle -> read -> compute -> write -> complete -> idle`
- `status.idle` and `status.running` must map correctly to FSM states

## Phase 3 - Register the Ball

Update files in order:
1. `.../configs/default.json` - append `ballIdMappings` entry and update `ballNum`
2. `.../bbus/busRegister.scala` - add import + `match case`
3. `.../DISA.scala` - add `val XXX = BitPat("bxxxxxxx")`
4. `.../DomainDecoder.scala` - add decode row (`BID = ballId.U`)

## Phase 4 - Add ISA C Macro

Create `<funct7_decimal>_<name>.c` under `bb-tests/workloads/lib/bbhw/isa/`, then include it in `isa.h`.

## Phase 5 - Add CTest

1. Create `<name>_test.c` under `bb-tests/workloads/src/CTest/toy/`
2. Register in `bb-tests/workloads/src/CTest/toy/CMakeLists.txt` using `add_cross_platform_test_target`
3. Append workload entry in `bb-tests/sardine/tests/test_ctest.py` (`ctest_workloads`)

## Phase 6 - Validate, Build, and Simulate

1. Run `validate` and ensure all 6 invariants pass
2. Run `bbdev_workload_build`
3. Run `bbdev_verilator_run` for this Ball's CTest binary
4. Interpret results:
   - `PASSED` -> done
   - `FAILED` -> switch to `/debug`
