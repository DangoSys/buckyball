# Buckyball Claude Code Workflow

Claude Code 作为交互前端，bbdev 作为执行后端。Claude 通过 MCP Server 调用 bbdev 的 HTTP API（server 模式，自动管理生命周期）。

## 三个 Workflow

| # | 触发 | 功能 |
|---|------|------|
| 1 | `/ball <Name>` | 新建 Ball：实现 → 注册 → ISA 宏 → CTest → 编译 → 仿真验证 |
| 2 | `/verify <Name>` | 验证 Ball：完整性检查 → 补全 → 编译 → 仿真 → 覆盖率分析 |
| 3 | `/optimize <Name>` | 优化 Ball：面积(yosys) + 时序(OpenSTA) + 延迟(仿真cycle) → 优化 → 回归验证 |

## 架构

```
用户 ──→ Claude Code (slash commands + CLAUDE.md)
              │
              ├── 读写代码：Read/Edit/Write
              ├── 静态校验：MCP validate
              └── 编译/仿真/综合/测试：MCP bbdev_* → bbdev HTTP API
                    │
                    └── bbdev server (Motia workflow 后端，MCP 自动管理生命周期)
                          ├── POST /verilator/run     全流程 clean→verilog→build→sim
                          ├── POST /verilator/verilog  生成 Verilog（支持 --balltype）
                          ├── POST /verilator/build    编译 Verilator（支持 --coverage）
                          ├── POST /verilator/sim      跑仿真（支持 --coverage）
                          ├── POST /workload/build     编译 CTest
                          ├── POST /sardine/run        批量测试（支持 --coverage → 覆盖率报告）
                          └── POST /yosys/synth        Yosys 综合 + OpenSTA 时序分析
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/claude/mcp_server.py` | MCP Server：validate + bbdev API 封装 + server 生命周期管理 |
| `.claude/settings.json` | MCP 配置 |
| `CLAUDE.md` | 全局指令：项目结构、Blink 协议、注册不变量、工具使用 |
| `.claude/commands/ball.md` | `/ball <Name>` 新建 Ball 全流程 |
| `.claude/commands/verify.md` | `/verify <Name>` 验证 Ball |
| `.claude/commands/optimize.md` | `/optimize <Name>` 优化 Ball |
| `.claude/commands/check.md` | `/check` 静态校验 |

## MCP Server 工具列表

### 校验
| 工具 | 功能 |
|------|------|
| `validate` | 检查 6 项注册不变量（ballId 递增/funct7 唯一/bid 对齐等） |

### bbdev API 封装
| 工具 | API | 说明 |
|------|-----|------|
| `bbdev_workload_build` | `/workload/build` | 编译 CTest |
| `bbdev_verilator_run` | `/verilator/run` | 全流程 clean→verilog→build→sim |
| `bbdev_verilator_verilog` | `/verilator/verilog` | 生成 Verilog |
| `bbdev_verilator_build` | `/verilator/build` | 编译 Verilator |
| `bbdev_verilator_sim` | `/verilator/sim` | 跑仿真 |
| `bbdev_sardine_run` | `/sardine/run` | 批量测试 |
| `bbdev_yosys_synth` | `/yosys/synth` | Yosys 综合 + OpenSTA |

## bbdev Server 生命周期

MCP Server 自动管理 bbdev server：
- 首次调用 bbdev_* 时自动启动（`pnpm dev --port <port>`）
- 启动前清理 BullMQ AOF 防止重放旧事件
- 端口从 5100-5500 自动选择
- 健康检查通过后才返回
- 每次调用前检查存活，挂了自动重启
- MCP Server 退出时自动清理

## Workflow 详细流程

### `/ball <Name>` — 新建 Ball

1. **需求收集**：读 default.json/DISA.scala 确定 ballId/funct7，问用户功能/inBW/outBW/op2
2. **实现 Ball**：参考现有 Ball 代码，在 prototype/ 下创建 wrapper/core/config
3. **注册**：更新 default.json + busRegister + DISA + DomainDecoder
4. **ISA 宏**：创建 C 宏文件，更新 isa.h
5. **CTest**：创建测试 .c，注册 CMakeLists.txt，追加 sardine 列表
6. **验证**：validate → bbdev_workload_build → bbdev_verilator_run → PASS/FAIL

### `/verify <Name>` — 验证 Ball

1. **完整性检查**：注册/ISA 宏/CTest/sardine 条目是否完整，缺什么补什么
2. **编译 + 仿真**：bbdev_workload_build → bbdev_verilator_run
3. **覆盖率分析**：bbdev_sardine_run(coverage=true) → 读覆盖率报告 → 建议补测试
4. **失败分析**：读仿真 log → 分析 Chisel 代码 → 提修复方案

### `/optimize <Name>` — 优化 Ball

1. **基线测量**：bbdev_yosys_synth（面积+时序）+ bbdev_verilator_run（cycle 数）
2. **面积分析**：从 hierarchy_report 提取子模块面积，识别面积大户
3. **时序/延迟分析**：timing_report 关键路径 + 仿真 cycle 数 + FSM 源码分析
4. **优化方案**：量化的方案列表（手段/面积变化/延迟变化/频率影响/trade-off）
5. **实施**：修改 Chisel 代码
6. **优化后测量**：再跑 yosys + verilator，输出前后对比报告
