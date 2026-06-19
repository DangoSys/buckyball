# BallCyclePMC

## TL;DR
- 【模块功能】旁路监听各 Ball 命令的下发与完成，按 `rob_id` 计算单次 `elapsed` 周期，并在完成时经 `DPI-C` 调用 `dpi_pmctrace`。
- 【模块定位】该模块是 `balldomain` 的 `bbus` 上的 `BallCyclePMC`，负责 Ball 性能计数。
- 【模块输入/输出】
    - Ball 性能计数：从单个 Ball 截取命令的下发与完成，计算 ball 的指令运行时间后经 `DPI-C` 输出到C代码仿真侧。
- 【关键点】
    - None

## Interface
| 方向 | 信号名 | 类型 | 含义 |
| --- | --- | --- | --- |
| 输入 | `cmdReq_i` | `Vec(numBalls, Valid(BallRsIssue))` | 第 `i` 路命令下发脉冲；`BBus` 将其 `valid` 接为 `cmdRouter.io.cmdReq_i(i).fire`。 |
| 输入 | `cmdResp_o` | `Vec(numBalls, Valid(BallRsComplete))` | 第 `i` 路命令完成指示；`BBus` 将其 `valid/bits` 接自 `cmdRouter.io.cmdResp_o(i)`。 |

`BallRsIssue` / `BallRsComplete` 中与计时相关的字段为 `rob_id`；`is_sub`、`sub_rob_id` 未参与本模块逻辑。

## Core Behavior
模块维护全局 `cycleCounter`，复位后每拍自增。`startTime` 长度为 `b.frontend.rob_entries`，按全局 `rob_id` 记录下发时刻。

对任意索引 `i`，当 `cmdReq_i(i).valid` 为真时，执行 `startTime(cmdReq_i(i).bits.rob_id) := cycleCounter`。当 `cmdResp_o(i).valid` 为真时，取 `robId = cmdResp_o(i).bits.rob_id`，计算 `elapsed = cycleCounter - startTime(robId)`。同一完成拍内，第 `i` 个 `PMCTraceDPI` 实例输出 `ball_id=i`、`rob_id=robId`、`elapsed=elapsed`，且 `enable=true`；其余周期各实例默认 `enable=false`。

模块内部实例化 `numBalls` 个 `PMCTraceDPI` BlackBox；其内联 Verilog 在 `enable` 为真时组合调用 `dpi_pmctrace(ball_id, rob_id, elapsed)`。

`BBus` 连接关系为：请求侧用握手成功 `fire` 作为起点脉冲，响应侧用 `valid` 作为完成脉冲；模块不参与 `ready` 门控，也不改变命令通路数据。

## Verification Checklist
- 复位后首拍 `cycleCounter` 应从 `0` 递增；若未自增判 fail。
- 在仅 `cmdReq_i(i).valid=1`、对应 `rob_id=X` 且尚未完成的场景，期望 `startTime(X)` 更新为当拍 `cycleCounter`；未更新判 fail。
- 在 `cmdReq_i(i).valid=1` 后经过 `N` 拍再拉高 `cmdResp_o(i).valid` 且 `rob_id` 仍为 `X`，期望当拍 `elapsed=N`（以起点拍与完成拍差值计）；偏差判 fail。
- 在 `cmdResp_o(i).valid=1` 的周期，期望对应 `PMCTraceDPI(i).enable=1` 且 `ball_id=i`、`rob_id`、`elapsed` 与内部计算一致；不一致或未触发判 fail。
- 在 `cmdResp_o(i).valid=0` 的周期，期望所有 `PMCTraceDPI.enable=0`；若仍调用 trace 判 fail。
- 对未出现过匹配 `cmdReq_i` 的 `rob_id` 直接注入 `cmdResp_o(i).valid=1`，模块仍会按 `cycleCounter - startTime(robId)` 计算；若需验证计时语义，应将此场景记为 fail 或单独标注为未定义起点行为。
- 两路及以上同拍分别完成时，期望各路 `PMCTraceDPI` 同拍独立触发；若丢失任一路完成事件判 fail。
