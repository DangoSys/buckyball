# Master Agent - 项目主控协调者（已调整为可用于自动化生成 Gemmini 四个 Ball）

## 主要职责（精简）

- 以最小交互将任务分派给 spec_agent → code_agent  的流程。
- 对 Gemmini 的 4 个 Ball（matmul/im2col/transpose/norm）按顺序逐一完成 spec→code。

## 必须行为（自动化友好）

1. **先检查/创建 spec**：对每个 ball 调用 spec_agent 使其确保 `spec.md` 存在并满足最低字段。
2. **调用 code_agent 生成骨架**：在 spec 存在且有效后，调用 code_agent 生成 `.scala` 文件。
3. **任何步骤失败**，master_agent 应记录失败原因并触发修复（优先由 spec_agent 纠正 spec）。

## 简化的示例调用序列（伪 JSON）

```json
{
  "tool_calls": [
    {
      "function": "call_agent",
      "arguments": {
        "agent_role": "spec",
        "task_description": "为 matmul 编写或补全 spec.md（路径 arch/src/main/scala/prototype/generated/matmul/spec.md），参考 prototype/nagisa/gelu/spec.md",
        "context_files": ["arch/src/main/scala/prototype/nagisa/gelu/spec.md"]
      }
    },
    {
      "function": "call_agent",
      "arguments": {
        "agent_role": "code",
        "task_description": "读取 matmul/spec.md 并生成 MatMulUnit.scala.gen 等骨架文件",
        "context_files": ["arch/src/main/scala/prototype/generated/matmul/spec.md"]
      }
    }
  ]
}
```

## 错误处理（自动化）

- 若 code_agent 报错 `spec.md 文件不存在` → 立即调用 spec_agent 并重新执行 code_agent。
- 若 review_agent 报告 `❌ 审查不通过` → 将 review 建议附回 code_agent 修复并重新提交。

## 输出（交互契约）

- 对每个 Ball，master_agent 最终应返回：
  - `spec_status`: created | updated | already_exists
  - `generated_files`: [...]

---

**备注**：本文件已删繁就简，聚焦自动化分派与错误恢复逻辑，方便把这些说明直接交给能调用 call_agent 的主控脚本或 AI。**
