# Buckyball

基于 RISC-V 的 DSA（Domain Specific Architecture）框架。Chisel 6.5.0，Nix Flake 构建。

## 项目结构

- `arch/src/main/scala/framework/` — 框架核心
  - `balldomain/prototype/` — Ball 算子实现（每个 Ball 一个子目录）
  - `balldomain/blink/` — Blink 协议定义（BlinkIO、BankRead/Write、BallStatus）
  - `balldomain/configs/` — BallDomainParam + default.json（ballIdMappings）
  - `balldomain/bbus/` — BBus 总线
  - `balldomain/rs/` — BallRsIssue / BallRsComplete（命令/完成接口）
  - `memdomain/backend/banks/` — SramReadIO / SramWriteIO
  - `core/bbtile/` — BBTile 集成（Rocket core + Buckyball）
  - `top/` — GlobalConfig（顶层参数汇聚）
- `arch/src/main/scala/examples/toy/balldomain/` — toy 配置
  - `DISA.scala` — 指令 opcode（funct7 BitPat）
  - `DomainDecoder.scala` — 指令解码表（ListLookup）
  - `bbus/busRegister.scala` — Ball 生成器注册（match case）
- `arch/src/main/scala/sims/` — 仿真配置
  - `verilator/` — Verilator 配置
  - `verify/` — 单 Ball elaboration（BallTopMain）
- `bb-tests/` — 测试
  - `workloads/lib/bbhw/isa/` — ISA C 宏（每条指令一个 .c 文件）
  - `workloads/src/CTest/toy/` — C 测试用例
  - `sardine/` — pytest 测试框架
- `bbdev/` — 开发工具链（Motia 工作流后端）

## Blink 协议

Ball 通过 Blink 协议接入 BBus。每个 Ball 实现 `HasBlink` trait。

```
BlinkIO(b: GlobalConfig, inBW: Int, outBW: Int):
  cmdReq:    Flipped(Decoupled(BallRsIssue))     // 命令输入（含 BallDecodeCmd + rob_id）
  cmdResp:   Decoupled(BallRsComplete)            // 完成输出（含 rob_id）
  bankRead:  Vec(inBW, Flipped(BankRead))         // SRAM 读端口
  bankWrite: Vec(outBW, Flipped(BankWrite))       // SRAM 写端口
  status:    BallStatus { idle, running }          // 状态信号

BankRead/BankWrite 元数据字段（均为 Input）:
  bank_id, rob_id, ball_id, group_id

SramReadIO:  req.valid/ready + req.bits.addr  →  resp.valid + resp.bits.data
SramWriteIO: req.valid/ready + req.bits(addr, data, mask, wmode)  →  resp.valid + resp.bits.ok
```

关键时序：SRAM 读延迟 = 1 cycle（req.fire 后下一周期 resp.valid 拉高）。

## 注册不变量

添加或修改 Ball 注册时，以下 6 项必须同时满足：

1. `default.json` 的 `ballNum` == `ballIdMappings` 数组长度
2. `ballId` 严格递增（0, 1, 2, ...），不跳号
3. `ballId` 无重复
4. `DISA.scala` 中 funct7 无重复
5. `busRegister.scala` 中的 case 名称集合 == `default.json` 中的 ballName 集合
6. `DomainDecoder.scala` 中的 BID 值集合 == `default.json` 中的 ballId 集合

用 `/check` 命令可以自动检查所有不变量。

## MCP 工具

项目配置了 `buckyball-dev` MCP Server，提供以下工具。

**重要：编译、仿真、综合、测试等操作必须通过 MCP 工具调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**
bbdev CLI 是给人类程序员用的，MCP 工具是给 agent 用的——MCP 工具内部会自动管理 bbdev server 生命周期并通过 HTTP API 调用。

### 校验
- `validate` — 检查 6 项注册不变量

### bbdev API 封装（自动管理 server 生命周期）
- `bbdev_workload_build` — 编译 CTest
- `bbdev_verilator_run` — 全流程 clean→verilog→build→sim
- `bbdev_verilator_verilog` — 生成 Verilog（支持 balltype 参数单独 elaborate 某个 Ball）
- `bbdev_verilator_build` — 编译 Verilator
- `bbdev_verilator_sim` — 跑仿真
- `bbdev_sardine_run` — 批量测试（支持 coverage）
- `bbdev_yosys_synth` — Yosys 综合 + OpenSTA 时序分析

### 分析报告路径
- 面积报告：`bbdev/api/steps/yosys/log/hierarchy_report.txt`（子模块分解）、`area_report.txt`（顶层）
- 时序报告：`bbdev/api/steps/yosys/log/timing_report.txt`
- 覆盖率报告：`bb-tests/sardine/reports/coverage/html/`
- 仿真日志：`arch/log/<timestamp>/stdout.log`、`disasm.log`

## 约定

- 改 Ball 实现时不要碰注册文件，改注册文件时不要碰实现文件
- Chisel 版本 6.5.0，不要使用 6.6+ 新 API
- CTest 用 `add_cross_platform_test_target` 注册到 CMakeLists.txt
- **禁止直接调用 `bbdev` CLI 或 `nix develop -c bbdev ...`**，必须通过 MCP 工具调用
- Ball wrapper 类名必须和 `default.json` 中 `ballName` 一致
