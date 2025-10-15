# Agent Demo - Gemmini NPU 开发

这个 demo 展示如何使用多 agent 协作开发一个兼容 Gemmini ISA 的 NPU 系统。

**📖 系统架构详解**: [ARCHITECTURE.md](./ARCHITECTURE.md) - Session 管理、多轮对话、Agent 通信机制

**⚠️ 代码保护规则**: [CODE_PROTECTION_RULES.md](./CODE_PROTECTION_RULES.md) - 现有代码保护，禁止删除/修改

**🔐 权限分配表**: [AGENT_PERMISSIONS.md](./AGENT_PERMISSIONS.md) - Agent 工具权限详细说明

**📁 工作范围规范**: [WORK_SCOPE.md](./WORK_SCOPE.md) - 工作路径限制、修改记录规范

## 系统架构

```
用户任务 (task/gemmini_npu.md)
    ↓
master_agent (主控协调)
    ↓
spec_agent (编写规范)
    ↓
code_agent (实现代码)
    ↓
review_agent (代码审查) ⭐ 新增
    ↓
verify_agent (测试验证)
```

## 使用方式

### 方式 1: API 调用

```bash
# 启动 master agent
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{
    "agentRole": "master",
    "promptPath": "workflow/steps/demo/prompt/task/gemmini_npu.md",
    "workDir": "/home/mio/Code/buckyball",
    "model": "deepseek-chat"
  }'
```

### 方式 2: 直接发送任务描述

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{
    "agentRole": "master",
    "promptPath": "<inline>",
    "promptContent": "实现一个兼容 Gemmini ISA 的 NPU 系统...",
    "workDir": "/home/mio/Code/buckyball"
  }'
```

## Agent 角色

### master_agent
- 任务：协调整体开发流程
- 输入：任务描述（task/*.md）
- 输出：项目规划和调度其他 agent
- **权限**：✅ 完全权限（可调用所有工具，包括 `call_agent` 和 `call_workflow_api`）

### spec_agent
- 任务：编写 Ball 的技术规范
- 输入：算子需求
- 输出：spec.md（参考 GELU spec）
- **权限**：文件操作 + Deepwiki（❌ 无编译/测试权限）

### code_agent
- 任务：实现并集成 Ball
- 输入：spec.md
- 输出：Chisel 代码 + ISA 定义 + 测试
- **前置检查**：必须先检查 spec.md 是否存在，否则停止并反馈给 master
- **执行顺序**：先完成 RTL（Chisel + ISA API + 系统注册），再编写测试用例
- **规则**：只添加新代码，不删除/修改已有代码
- **权限**：文件操作 + Deepwiki（❌ 无编译/测试权限）

### review_agent ⭐ 新增
- 任务：审查代码完整性和质量
- 输入：code_agent 的实现
- 输出：PASS（通过）或 FAIL（问题列表 + 修复建议）
- **审查顺序**：优先检查 RTL 是否完整，再检查测试用例
- **重点**：检查是否删除/修改了已有代码、RTL 未完成就写测试
- **权限**：文件读取 + 搜索（❌ 无编译/测试权限）

### verify_agent
- 任务：测试验证
- 输入：review 通过的代码
- 输出：测试报告 + verilator 仿真结果
- **规则**：只运行测试，不修改代码
- **权限**：文件操作 + ✅ **Workflow API**（编译、测试）

## 可用工具与权限

**🔐 详细权限说明**: 请参考 [AGENT_PERMISSIONS.md](./AGENT_PERMISSIONS.md)

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
