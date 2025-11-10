# Gemmini Ball Generator - 简化版 Agent Demo

这个 demo 展示如何使用**单一智能 Agent**自动生成 Gemmini NPU 的 4 个 Ball（MatMul, Im2col, Transpose, Norm）。

> **🎯 新版本特点**：从复杂的多 Agent 协作改为**单一 Agent** 自动完成所有工作，更简单、更可靠。

## 🚀 快速开始

### 方式 1: 直接运行 Python（推荐）

```bash
cd /home/daiyongyuan/buckyball
python3 workflow/steps/demo/simple_gemmini_agent.py
```

### 方式 2: 使用启动脚本

```bash
# 直接模式
bash workflow/steps/demo/test_demo.sh

# 或使用 API 模式（需要 bbdev 服务）
bash workflow/steps/demo/test_demo.sh api
```

## 📖 详细文档

- **快速开始**: [/GEMMINI_QUICKSTART.md](/GEMMINI_QUICKSTART.md)
- **完整文档**: [prompt/README.md](./prompt/README.md)
- **Agent 指令**: [prompt/gemmini_ball_generator.md](./prompt/gemmini_ball_generator.md)
- **任务描述**: [prompt/gemmini_task.md](./prompt/gemmini_task.md)

## 🏗️ 新架构（简化版）

```
Gemmini Ball Generator (单一智能 Agent)
├─ 学习阶段: 读取参考代码 (VecUnit, VecBall)
├─ 生成循环 (4次):
│  ├─ 生成代码 (Unit.scala + Ball.scala)
│  ├─ 更新注册 (DomainDecoder, busRegister, rsRegister, DISA)
│  ├─ 编译验证 (build_gemmini.sh)
│  └─ 错误修复 (自动分析并修复，最多5次)
└─ 完成条件: 所有 4 个 Ball 编译成功
```

### 对比旧架构

| 特性 | 旧架构（多Agent） | 新架构（单Agent） |
|------|------------------|------------------|
| Agent 数量 | 5个 | **1个** |
| 复杂度 | 高 | **低** |
| 停止问题 | ❌ 经常停止 | ✅ **自动持续** |
| 错误恢复 | 分散 | ✅ **统一修复** |
| 代码量 | ~1500行 | **~350行** |

## 🎯 Agent 能力

单一 Agent 具备完整能力：

1. **学习能力** - 自动读取并理解参考代码
2. **生成能力** - 生成完整可编译的 Chisel 代码
3. **验证能力** - 自动调用编译脚本验证
4. **修复能力** - 智能分析编译错误并自动修复
5. **持续能力** - 自动完成所有 4 个 Ball

## 🛠️ 可用工具

Agent 可以使用以下工具（在 `simple_gemmini_agent.py` 中实现）：

| 工具 | 功能 | 说明 |
|-----|------|------|
| `read_file` | 读取文件内容 | 支持相对路径和绝对路径 |
| `write_file` | 写入文件内容 | 自动创建父目录 |
| `list_files` | 列出目录文件 | 返回文件列表 |
| `make_dir` | 创建目录 | 递归创建（mkdir -p） |
| `run_build` | 运行编译验证 | 自动调用 build_gemmini.sh 并分析结果 |
| `grep_files` | 搜索文件内容 | 使用 grep 搜索模式 |

## 📊 执行流程

```
[迭代 1] 🔧 读取参考代码 (VecUnit.scala, VecBall.scala)
[迭代 2] 🔧 生成 MatMulUnit.scala
[迭代 3] 🔧 生成 MatMulBall.scala
[迭代 4] 🔧 更新系统注册文件
[迭代 5] 🔧 运行编译
           ✅ 编译成功
           ✅ MatMul Ball 完成！
[迭代 6] 🔧 开始生成 Im2col...
...
[迭代 N] 🎉 所有 4 个 Ball 完成！
```

## 🔍 旧文档（已废弃）

以下文档仍然保留，但**仅供参考**，不再使用：

- ~~[00_code_agent_event_step.py](./00_code_agent_event_step.py)~~ - 旧的事件驱动系统
- ~~[00_code_agent_api_step.py](./00_code_agent_api_step.py)~~ - 旧的 API 步骤
- ~~[ARCHITECTURE.md](./ARCHITECTURE.md)~~ - 旧的多 Agent 架构
- ~~[AGENT_PERMISSIONS.md](./AGENT_PERMISSIONS.md)~~ - 旧的权限系统
- ~~[CODE_PROTECTION_RULES.md](./CODE_PROTECTION_RULES.md)~~ - 代码保护规则
- ~~[WORK_SCOPE.md](./WORK_SCOPE.md)~~ - 工作范围规范

**新系统**只需要 `simple_gemmini_agent.py` 和两个 prompt 文件。

## 📁 生成的文件

成功执行后，会在以下位置生成代码：

```
arch/src/main/scala/prototype/gemmini/
├── matmul/
│   ├── MatMulUnit.scala
│   └── MatMulBall.scala
├── im2col/
│   ├── Im2colUnit.scala
│   └── Im2colBall.scala
├── transpose/
│   ├── TransposeUnit.scala
│   └── TransposeBall.scala
└── norm/
    ├── NormUnit.scala
    └── NormBall.scala
```

同时会更新系统注册文件：
- `examples/toy/balldomain/DomainDecoder.scala`
- `examples/toy/balldomain/busRegister.scala`
- `examples/toy/balldomain/rsRegister.scala`
- `examples/toy/balldomain/DISA.scala`

## 🐛 故障排查

### Agent 停止执行
检查是否达到最大迭代次数（默认100次），可在 `simple_gemmini_agent.py` 中调整。

### 编译失败无法修复
查看编译日志：`/home/daiyongyuan/buckyball/build_logs/gemmini_build.log`

### API 调用失败
检查 `.env` 配置，确保 LLM API 可访问。

## 🎓 设计理念

**简单优于复杂** - 单一 Agent 自动完成所有工作，无需复杂的协作机制。

## 可用工具与权限

**🔐 详细权限说明**: 新系统中 Agent 拥有所有必需的工具权限。

### 工具权限矩阵

| 工具类型 | Master | Spec | Code | Review | Verify |
|---------|--------|------|------|--------|--------|
| 文件操作 | ✅ 完全 | ✅ 完全 | ✅ 完全 | ✅ 读取 | ✅ 完全 |
| Deepwiki | ✅ | ✅ | ✅ | ❌ | ❌ |
| `call_agent` | ✅ 独有 | ❌ | ❌ | ❌ | ❌ |
| `call_workflow_api` | ✅ | ❌ | ❌ | ❌ | ✅ |

### 文件操作工具（所有 agent 可用）
- `read_file`: 读取文件
- `write_file`: 写入文件
- `list_files`: 列出目录
- `make_dir`: 创建目录
- `delete_file`: 删除文件
- `grep_files`: 搜索文件内容

### Deepwiki 工具（spec/code/master 可用）
- `deepwiki_ask`: 询问仓库问题
  - repo: "DangoSys/buckyball" 或 "ucb-bar/gemmini"
  - question: 你的问题
- `deepwiki_read_wiki`: 读取仓库文档

### Agent 协调工具（仅 master 可用）
- `call_agent`: 调用其他 agent
  - agent_role: "spec" | "code" | "review" | "verify"
  - task_description: 任务说明
  - context_files: 上下文文件路径（可选）

### Workflow API 工具（仅 master 和 verify 可用）
- `call_workflow_api`: 调用内部 workflow API
  - `/verilator/verilog`: 生成 Verilog
  - `/verilator/build`: 编译 verilator (params: jobs)
  - `/verilator/sim`: 运行仿真 (params: binary, batch)
  - `/workload/build`: 编译测试程序 (params: args)
  - `/sardine/run`: 运行 sardine 测试 (params: workload)

## 工作流程示例

1. Master agent 读取 `gemmini_npu.md`
2. Master 用 Deepwiki 查询 Gemmini 和 ToyBuckyBall
3. Master 规划需要实现的 Ball 列表
4. 对每个 Ball：
   - 调用 spec_agent 编写 spec
   - 调用 code_agent 实现（**先完成 RTL，再写测试**）⭐
   - 调用 review_agent 审查（优先检查 RTL 完整性）⭐
   - 如果审查通过，调用 verify_agent 测试
   - 如果审查不通过（RTL 未完成/流程错误），code_agent 修复后重新审查
5. Master 集成所有 Ball 成 NPU 系统
6. 端到端测试验证

## 预期输出

```
arch/src/main/scala/prototype/gemmini/
├── dma/
│   ├── spec.md
│   ├── DMAUnit.scala
│   └── ...
├── matmul/
│   ├── spec.md
│   ├── MatMulUnit.scala
│   └── ...
└── ...

arch/src/main/scala/examples/gemmini/
├── GemminiNPU.scala
├── DomainDecoder.scala
└── ...

bb-tests/workloads/src/CTest/
├── gemmini_mvin_test.c
├── gemmini_matmul_test.c
└── ...

docs/
├── plan.md
├── summary.md
└── test_report.md
```

## 注意事项

1. **环境要求**：
   - Python 3.8+
   - 配置好 API_KEY 和 BASE_URL
   - MCP 服务器运行中

2. **开发顺序**：
   - 先实现核心 Ball（DMA、MatMul）
   - 再扩展高级功能（Loop 指令）

3. **代码保护规则** ⭐ 重要：
   - **现有代码是正确的，只添加不修改**
   - 不要删除任何已有代码
   - 不要修改已有 Ball 实现
   - 只在指定位置追加新代码
   - 详见：[CODE_PROTECTION_RULES.md](./CODE_PROTECTION_RULES.md)

4. **调试技巧**：
   - 查看 session 日志
   - 使用 Deepwiki 查询不懂的内容
   - 参考现有 Ball 实现
