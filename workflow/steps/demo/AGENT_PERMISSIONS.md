# Agent 权限分配表

## 权限矩阵

| Agent | 文件操作 | Deepwiki | call_agent | call_workflow_api |
|-------|---------|----------|------------|-------------------|
| **master_agent** | ✅ 完全 | ✅ | ✅ 独有 | ✅ |
| **spec_agent** | ✅ 完全 | ✅ | ❌ | ❌ |
| **code_agent** | ✅ 完全 | ✅ | ❌ | ❌ |
| **review_agent** | ✅ 读写 | ❌ | ❌ | ❌ |
| **verify_agent** | ✅ 完全 | ❌ | ❌ | ✅ |

## 详细权限说明

### Master Agent（项目主控）
**最高权限 - 可以使用所有工具**

✅ **独有权限：**
- `call_agent`: 调用其他 agent（spec/code/review/verify）

✅ **完整权限：**
- 文件操作：`read_file`, `write_file`, `list_files`, `make_dir`, `delete_file`, `grep_files`
- Deepwiki：`deepwiki_ask`, `deepwiki_read_wiki`
- Workflow API：`call_workflow_api`（编译、测试）

**职责：**
- 协调整体开发流程
- 调度其他 agent
- 必要时可以直接操作所有工具

---

### Spec Agent（规格书编写）
**文档编写权限 - 只能读写文件和查询文档**

✅ **允许的工具：**
- 文件操作：`read_file`, `write_file`, `list_files`, `make_dir`
- 搜索：`grep_files`
- Deepwiki：`deepwiki_ask`, `deepwiki_read_wiki`

❌ **禁止的工具：**
- `call_agent`：不能调用其他 agent
- `call_workflow_api`：不能编译或测试

**职责：**
- 编写 spec.md
- 查询参考文档
- 定义 Ball 规格

**设计理由：**
Spec agent 只负责文档编写，不需要运行代码或测试，给予编译/测试权限会增加误操作风险。

---

### Code Agent（代码实现）
**代码编写权限 - 只能读写文件和查询文档**

✅ **允许的工具：**
- 文件操作：`read_file`, `write_file`, `list_files`, `make_dir`, `delete_file`
- 搜索：`grep_files`
- Deepwiki：`deepwiki_ask`, `deepwiki_read_wiki`

❌ **禁止的工具：**
- `call_agent`：不能调用其他 agent
- `call_workflow_api`：不能编译或测试

**职责：**
- 实现 Chisel RTL 代码
- 定义 ISA API
- 注册 Ball 到系统
- 编写测试用例

**设计理由：**
Code agent 专注于代码实现，不应该自己运行编译或测试（这是 verify_agent 的工作）。分离实现和验证职责，避免 code agent 在代码未完成时就尝试运行测试。

**前置检查：**
Code agent 必须先检查 spec.md 是否存在，如果不存在应该停止并反馈给 master_agent。

---

### Review Agent（代码审查）
**审查权限 - 只能读文件，必要时可以修复小问题**

✅ **允许的工具：**
- 文件操作：`read_file`, `list_files`, `grep_files`
- 修复小问题：`write_file`（谨慎使用）

❌ **禁止的工具：**
- `call_agent`：不能调用其他 agent
- `call_workflow_api`：不能编译或测试
- `deepwiki_ask`：不需要查询文档

**职责：**
- 审查代码完整性
- 检查 RTL 是否完成
- 检查是否删除/修改了已有代码
- 必要时修复小问题（如格式、拼写）

**设计理由：**
Review agent 专注于审查，不需要运行代码或查询文档。主要使用读取工具检查代码质量。

---

### Verify Agent（测试验证）
**测试执行权限 - 可以运行编译和测试**

✅ **允许的工具：**
- 文件操作：`read_file`, `write_file`, `list_files`, `make_dir`
- 搜索：`grep_files`
- **Workflow API**：`call_workflow_api` ✅

**Workflow API 详细权限：**
- `/verilator/verilog`：生成 Verilog
- `/verilator/build`：编译 Verilator
- `/verilator/sim`：运行仿真
- `/workload/build`：编译测试程序
- `/sardine/run`：运行回归测试

❌ **禁止的工具：**
- `call_agent`：不能调用其他 agent
- `deepwiki_ask`：不需要查询文档

**职责：**
- 编译测试程序
- 运行 Verilator 仿真
- 执行 Sardine 回归测试
- 生成测试报告

**设计理由：**
Verify agent 是唯一允许运行编译和测试的 agent（除了 master_agent）。这样确保测试流程集中管理，避免其他 agent 在不合适的时机运行测试。

---

## 权限分配原则

### 1. 最小权限原则
每个 agent 只拥有完成其职责所需的最小权限集合。

### 2. 职责分离
- **Spec Agent**：只写文档
- **Code Agent**：只写代码
- **Review Agent**：只审查
- **Verify Agent**：只测试
- **Master Agent**：协调所有

### 3. 安全考虑

**为什么 code_agent 和 spec_agent 不能有 workflow API 权限？**

1. **防止过早测试**：避免 code_agent 在 RTL 未完成时就运行测试
2. **流程控制**：确保测试只在 review 通过后由 verify_agent 执行
3. **职责明确**：代码实现和测试验证分离，便于 debug
4. **减少误操作**：避免 spec_agent 意外触发编译

**为什么只有 master_agent 可以调用其他 agent？**

1. **流程控制**：master_agent 作为唯一的协调者，确保工作流程正确
2. **避免循环调用**：防止 agent 之间互相调用形成死循环
3. **清晰层级**：master → spec/code/review/verify 单向调用

## 权限检查机制

### 在 Agent Prompt 中明确声明

每个 agent 的 prompt 都应该包含：

```markdown
## 可用工具
- [列出允许的工具]

## ⚠️ 无权限工具
- ❌ [列出禁止的工具及原因]
```

### 工具调用时的权限检查

如果 agent 尝试调用无权限的工具，应该：

1. **拒绝执行**：返回错误信息
2. **明确提示**：告知该工具需要哪个 agent 才能使用
3. **建议方案**：如何通过 master_agent 调用有权限的 agent

## 权限变更记录

### 2025-01-15
- 移除 code_agent 的 `call_workflow_api` 权限
- 移除 code_agent 的 `call_agent` 权限
- 移除 spec_agent 的所有高级权限（仅保留文件操作和 Deepwiki）
- 明确 verify_agent 拥有 `call_workflow_api` 权限
- 明确 master_agent 拥有所有工具的权限
- 添加 code_agent 前置检查：必须先检查 spec.md 是否存在

## 总结

**权限层级（从高到低）：**

1. **Master Agent**：完全权限
2. **Verify Agent**：文件操作 + Workflow API
3. **Spec Agent / Code Agent**：文件操作 + Deepwiki
4. **Review Agent**：文件操作（主要是读取）

**核心原则：谁负责，谁有权限。不负责的事情，不给权限。**
