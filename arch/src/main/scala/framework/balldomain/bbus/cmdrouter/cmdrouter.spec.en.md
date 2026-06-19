# CmdRouter

## TL;DR
- 【Module Function】Arbitrates `numBalls` request lanes from `cmdReq_i` into one `cmdReq_o`, and forwards each `cmdResp_i` lane to the matching `cmdResp_o` lane.
- 【Module Placement】This module is `CmdRouter` on the `bbus` in `balldomain`, responsible for command routing.
- 【Module Inputs/Outputs】
    - Command dispatch: `cmdReq_i` from frontend to bbus; after round-robin arbitration, single `cmdReq_o` to one ball.
    - Command completion response: `cmdResp_i` from each ball; forwarded lane-by-lane through `cmdResp_o` to frontend.
    - BBus drives `ballIdle` to report whether each ball is idle.
- 【Key Points】
    - Request arbitration uses registered `ballIdleR` only, so `ballIdle` affects admission with a one-cycle delay.

## Interface
| Direction | Signal | Type | Meaning |
| --- | --- | --- | --- |
| Input | `cmdReq_i` | `Vec(numBalls, Flipped(Decoupled(BallRsIssue)))` | Per-Ball request ingress channels using `valid/ready` handshake. |
| Input | `cmdResp_i` | `Vec(numBalls, Flipped(Decoupled(BallRsComplete)))` | Per-Ball response ingress channels, forwarded lane by lane. |
| Input | `ballIdle` | `Vec(numBalls, Bool)` | Per-Ball idle indicators, registered before request gating. |
| Output | `cmdReq_o` | `Decoupled(BallRsIssue)` | Single arbitrated request egress channel. |
| Output | `cmdResp_o` | `Vec(numBalls, Decoupled(BallRsComplete))` | Response egress channels directly mapped to `cmdResp_i` by index. |

## Core Behavior
The module instantiates `RRArbiter(BallRsIssue, numBalls)` for round-robin request selection and computes `ballIdleR = RegNext(io.ballIdle, false)` as the lane-enable condition. For any index `i`, a request is presented to arbitration only when both `cmdReq_i(i).valid` and `ballIdleR(i)` are true. The same `ballIdleR(i)` condition gates `cmdReq_i(i).ready`, so upstream always sees `ready = false` when that lane is not idle in the registered view. The arbiter output is directly connected to `cmdReq_o` with `<>`, so `valid/bits/ready` semantics follow the standard `RRArbiter` plus Decoupled connection behavior. The response path performs no selection or transformation: each lane uses `cmdResp_o(i) <> cmdResp_i(i)` and keeps independent handshakes.

## Verification Checklist
- With all `ballIdle` lanes held at `0` and all `cmdReq_i.valid` held at `1`, `cmdReq_o.valid` must remain `0`; any observed `1` is fail.
- Raise one `ballIdle(i)` from `0` to `1` at cycle `t` while keeping `cmdReq_i(i).valid=1`; that lane may be admitted no earlier than cycle `t+1`; admission at `t` is fail.
- With `ballIdle(i)=0` and `cmdReq_i(i).valid=1`, `cmdReq_i(i).ready` must stay `0`; any cycle with `ready=1` is fail.
- With two or more continuously valid lanes and their `ballIdleR` all `1`, `cmdReq_o.bits` source should rotate under round-robin instead of sticking forever to one lane; persistent starvation is fail.
- Inject a response transaction on any lane `i` while driving `cmdResp_o(i).ready=1`; `cmdResp_o(i).valid` and `bits` must match input in the same cycle; mismatch is fail.
- Drive different response streams on lanes `i` and `j (i!=j)` concurrently; outputs must remain lane-isolated with no cross-lane mixing; any mixing is fail.
- When downstream forces `cmdReq_o.ready=0`, request handshakes must stop and no fake completion can occur; any observed `fire` is fail.
- Toggle `ballIdle` across adjacent cycles and verify gating always follows one-cycle-late `RegNext` behavior; same-cycle pass-through is fail.
