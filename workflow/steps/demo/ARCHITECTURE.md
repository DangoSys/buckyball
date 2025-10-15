# Agent 系统架构

## 多轮对话机制

### Session 管理

每个 agent 运行在**独立的 session** 中：

```
User → Master Agent (Session A)
         ↓
         ├─ call_agent("spec") → Spec Agent (Session B, 独立)
         ├─ call_agent("code") → Code Agent (Session C, 独立)
         └─ call_agent("verify") → Verify Agent (Session D, 独立)
```

### Session Store

- **存储**: Redis（生产）或内存（开发）
- **TTL**: 24 小时（可配置）
- **内容**: 完整的消息历史（system/user/assistant/tool）

```python
# 创建新 session
POST /agent {"agentRole": "master", "promptPath": "task.md"}

# 继续已有 session
POST /agent {"agentRole": "master", "promptPath": "followup.md", "sessionId": "xxx"}
```

### Agent 间通信

Master 通过 `call_agent` 工具调用其他 agent：

```python
# Master 调用 Spec Agent
call_agent(
  agent_role="spec",
  task_description="为 ReluBall 编写 spec.md",
  context_files=["arch/src/.../gelu/spec.md"]
)

# 返回值
{
  "status": "success",
  "agent": "spec",
  "result": {...},
  "files": ["arch/.../relu/spec.md"]
}
```

### 调用链

```
1. Master (Session A):
   - 读取 gemmini_npu.md
   - 使用 deepwiki_ask 查询 Gemmini 架构
   - 规划需要实现的 Ball 列表

2. Master → Spec Agent (新 Session B):
   - 创建临时任务文件 .agent_tasks/task_spec_xxx.md
   - POST /agent {"agentRole": "spec", ...}
   - Spec Agent 独立运行，生成 spec.md
   - 返回结果给 Master

3. Master → Code Agent (新 Session C):
   - 传递 spec.md 路径作为 context
   - Code Agent 实现 Chisel 代码 + ISA API

4. Code Agent → Verify Agent (新 Session D):
   - 传递测试文件路径
   - Verify Agent 运行 verilator 仿真
   - 返回测试结果

5. Master 收集所有结果:
   - 检查文件生成
   - 整合系统
   - 返回最终报告
```

## 工具能力矩阵

| 工具 | Master | Spec | Code | Verify |
|------|--------|------|------|--------|
| `call_agent` | ✅ | ❌ | ✅ (仅 verify) | ❌ |
| `deepwiki_ask` | ✅ | ✅ | ✅ | ❌ |
| `deepwiki_read_wiki` | ✅ | ✅ | ✅ | ❌ |
| `call_workflow_api` | ✅ | ❌ | ✅ | ✅ |
| 文件操作 | ✅ | ✅ | ✅ | ✅ |

## 数据流

```
gemmini_npu.md (User Task)
    ↓
Master Agent
    ↓
┌───┴───┬───────┬──────────┐
↓       ↓       ↓          ↓
DMABall ConfigBall MatMulBall LoopBall ...
│       │       │          │
Spec    Spec    Spec       Spec
│       │       │          │
Code    Code    Code       Code
│       │       │          │
Verify  Verify  Verify     Verify
│       │       │          │
└───┬───┴───────┴──────────┘
    ↓
System Integration (Master)
    ↓
End-to-End Test
    ↓
NPU Ready!
```

## Session 恢复

如果需要继续之前的任务：

```bash
# 第一次调用
curl -X POST http://localhost:3001/agent \
  -d '{"agentRole": "master", "promptPath": "task.md"}'
# 返回: {"sessionId": "abc123", ...}

# 继续对话
curl -X POST http://localhost:3001/agent \
  -d '{"agentRole": "master", "promptPath": "followup.md", "sessionId": "abc123"}'
```

## 工具实现

所有工具在 `workflow/steps/tools/` 中：

- `file_tools.py`: 文件操作（7 个工具）
- `agent_tools.py`: Agent 协调（`call_agent`）
- `workflow_tools.py`: Workflow API（`call_workflow_api`）
- `deepwiki_tools.py`: Deepwiki 集成（2 个工具）

## 错误处理

- **子 agent 失败**: 返回错误给父 agent，由父 agent 决定如何处理
- **工具调用失败**: 返回 JSON 错误信息，LLM 可以重试或调整
- **超时**: httpx 设置 timeout，防止无限等待
- **迭代限制**: 每个 agent 最多 10 轮迭代，防止死循环

## 安全考虑

- **临时文件**: 创建在 `.agent_tasks/` 目录，任务完成后清理
- **路径验证**: 所有文件操作基于 `work_dir`，防止越权访问
- **工具限制**: 不同 agent 只能使用允许的工具
- **Session TTL**: 24 小时自动过期，避免内存泄漏
