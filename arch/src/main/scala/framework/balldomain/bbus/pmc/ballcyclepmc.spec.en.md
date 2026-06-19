# BallCyclePMC

## TL;DR
- 【Module Function】Passively monitors command issue and completion for each Ball, computes per-command `elapsed` cycles by `rob_id`, and calls `dpi_pmctrace` via DPI-C on completion.
- 【Module Placement】This module is `BallCyclePMC` on the `bbus` in `balldomain`, responsible for Ball performance counting.
- 【Module Inputs/Outputs】
    - Ball performance counting: taps command issue and completion on a single Ball, computes per-ball instruction runtime, and outputs to the C-side simulator via DPI-C.
- 【Key Points】
    - None

## Interface
| Direction | Signal | Type | Meaning |
| --- | --- | --- | --- |
| Input | `cmdReq_i` | `Vec(numBalls, Valid(BallRsIssue))` | Lane `i` command-issue pulse; `BBus` drives `valid` from `cmdRouter.io.cmdReq_i(i).fire`. |
| Input | `cmdResp_o` | `Vec(numBalls, Valid(BallRsComplete))` | Lane `i` command-completion indicator; `BBus` drives `valid/bits` from `cmdRouter.io.cmdResp_o(i)`. |

Only `rob_id` in `BallRsIssue` / `BallRsComplete` participates in timing; `is_sub` and `sub_rob_id` are unused.

## Core Behavior
The module keeps a global `cycleCounter` that increments every cycle after reset. `startTime` has length `b.frontend.rob_entries` and records the issue time per global `rob_id`.

For any index `i`, when `cmdReq_i(i).valid` is true, it executes `startTime(cmdReq_i(i).bits.rob_id) := cycleCounter`. When `cmdResp_o(i).valid` is true, it takes `robId = cmdResp_o(i).bits.rob_id`, computes `elapsed = cycleCounter - startTime(robId)`, and in the same completion cycle drives `PMCTraceDPI` instance `i` with `ball_id=i`, `rob_id=robId`, `elapsed=elapsed`, and `enable=true`; all instances default to `enable=false` in other cycles.

The module instantiates `numBalls` `PMCTraceDPI` BlackBoxes; the inlined Verilog combinationally calls `dpi_pmctrace(ball_id, rob_id, elapsed)` when `enable` is true.

`BBus` wiring uses handshake `fire` on the request side as the start pulse and `valid` on the response side as the completion pulse; the module does not gate `ready` and does not alter command-path data.

## Verification Checklist
- After reset, `cycleCounter` must increment from `0` on the first cycle; failure to increment is fail.
- With only `cmdReq_i(i).valid=1`, matching `rob_id=X`, and no completion yet, `startTime(X)` must update to the current `cycleCounter`; failure to update is fail.
- After `cmdReq_i(i).valid=1`, wait `N` cycles, then assert `cmdResp_o(i).valid` with the same `rob_id=X`; `elapsed` in that cycle must be `N` (difference between issue and completion cycles); any deviation is fail.
- When `cmdResp_o(i).valid=1`, `PMCTraceDPI(i).enable` must be `1` and `ball_id`, `rob_id`, and `elapsed` must match internal computation; mismatch or no trigger is fail.
- When `cmdResp_o(i).valid=0`, all `PMCTraceDPI.enable` must be `0`; any trace call is fail.
- Inject `cmdResp_o(i).valid=1` for a `rob_id` that never had a matching `cmdReq_i`; the module still computes `cycleCounter - startTime(robId)`; treat as fail or separately mark as undefined start behavior when validating timing semantics.
- When two or more lanes complete in the same cycle, each `PMCTraceDPI` must fire independently in that cycle; losing any completion event is fail.
