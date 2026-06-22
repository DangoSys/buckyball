# CmdRouter

## TL;DR
- 【模块功能】将 `numBalls` 路 `cmdReq_i` 按轮询仲裁汇聚到单一路 `cmdReq_o`，并把每路 `cmdResp_i` 一一透传到对应的 `cmdResp_o`。
- 【模块定位】该模块是 `balldomain`的`bbus`上的 `CmdRouter` 负责指令路由`。
- 【模块输入/输出】
    - 指令分发：从前端到bbus输入 `cmdReq_i`，选择后输出单路 `cmdReq_o` 给某个ball。
    - 指令（完成后）响应：从每个ball输入`cmdResp_o` 的响应信号，选择后输出单路给前端。
    - bbus通过 `ballIdle` 获取每个ball是否空闲
- 【关键点】
    - 请求仲裁只看寄存后的 `ballIdleR`

## Interface
| 方向 | 信号名 | 类型 | 含义 |
| --- | --- | --- | --- |
| 输入 | `cmdReq_i` | `Vec(numBalls, Flipped(Decoupled(BallRsIssue)))` | 来自各 Ball 的请求输入通道，按 `valid/ready` 握手。 |
| 输入 | `cmdResp_i` | `Vec(numBalls, Flipped(Decoupled(BallRsComplete)))` | 来自各 Ball 的响应输入通道，逐路透传到输出。 |
| 输入 | `ballIdle` | `Vec(numBalls, Bool)` | 各 Ball 的空闲状态输入，先经 `RegNext` 后参与请求门控。 |
| 输出 | `cmdReq_o` | `Decoupled(BallRsIssue)` | 仲裁后的单一路请求输出通道。 |
| 输出 | `cmdResp_o` | `Vec(numBalls, Decoupled(BallRsComplete))` | 与 `cmdResp_i` 同索引直连的响应输出通道。 |

## Core Behavior
模块内部实例化 `RRArbiter(BallRsIssue, numBalls)` 对请求进行轮询仲裁，并使用 `ballIdleR = RegNext(io.ballIdle, false)` 作为每路请求的使能条件。对任意索引 `i`，只有当 `cmdReq_i(i).valid` 与 `ballIdleR(i)` 同时为真时，该路才会向仲裁器宣告有效请求；同一条件也用于反向门控 `cmdReq_i(i).ready`，因此当 `ballIdleR(i)` 为假时，上游看到的 `ready` 必为假。仲裁器输出通过 `<>` 直接连接到 `cmdReq_o`，其 `valid/bits/ready` 握手语义由 `RRArbiter` 与该直连共同决定。响应路径不做选择与变换，每个索引 `i` 上 `cmdResp_o(i) <> cmdResp_i(i)`，保持逐路独立握手。

## Verification Checklist
- 在 `ballIdle` 全为 `0` 且各路 `cmdReq_i.valid` 为 `1` 的场景，期望 `cmdReq_o.valid` 始终为 `0`；若出现 `1` 判 fail。
- 在第 `t` 周期将某一路 `ballIdle(i)` 从 `0` 拉到 `1`，并保持 `cmdReq_i(i).valid=1`，期望最早在 `t+1` 周期该路才可能被仲裁；若 `t` 周期已被放行判 fail。
- 在某一路 `ballIdle(i)=0` 且 `cmdReq_i(i).valid=1` 时，检查 `cmdReq_i(i).ready` 必为 `0`；任一周期为 `1` 判 fail。
- 在两路及以上持续有效且对应 `ballIdleR` 均为 `1` 的场景，观察 `cmdReq_o.bits` 来源应随轮询发生切换，不应永久固定在同一路；若长期饥饿某一路判 fail。
- 对任意索引 `i` 注入 `cmdResp_i(i)` 事务并驱动 `cmdResp_o(i).ready=1`，期望同拍 `cmdResp_o(i).valid` 与 `bits` 等于输入；不一致判 fail。
- 对任意索引 `i/j(i!=j)` 同时施加不同响应流量，期望两路 `cmdResp_o` 互不串扰；若出现跨索引数据混入判 fail。
- 在下游将 `cmdReq_o.ready` 拉低时，期望仲裁输出握手暂停且不会伪造完成；若无 `ready` 仍出现 `fire` 判 fail。
- 将 `ballIdle` 在相邻周期内反复翻转，验证门控行为始终按 `RegNext` 延后一拍；若出现当拍直通效果判 fail。
