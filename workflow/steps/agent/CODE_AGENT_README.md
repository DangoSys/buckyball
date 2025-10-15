# Code Agent Step 使用说明

## 概述
Code Agent Step 是一个基于 AI 的智能代码助手，使用 **Function Calling** 技术让 AI 自主决定如何读取和生成代码。

### ✨ 核心特性
- 🤖 **智能决策**：AI 自己决定先读哪些文件，再决定如何生成代码
- 🔄 **多轮对话**：支持会话管理，AI 可以记住之前的操作
- 🛠️ **工具调用**：AI 可以使用 read_file、write_file、list_files 等工具
- 📝 **自动迭代**：AI 可以多次读写文件，直到完成任务

## 文件结构
- `02_code_agent_api_step.py` - API 入口，接收请求
- `02_code_agent_event_step.py` - 事件处理器，执行 AI 调用和文件操作
- `example_prompt.md` - 示例 prompt 文件

## API 接口

### 端点
```
POST /agent/code
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|--------|
| promptPath | string | 是 | Markdown prompt 文件路径 | - |
| workDir | string | 否 | 工作目录路径 | 当前目录 |
| model | string | 否 | AI 模型名称 | deepseek-chat |
| apiKey | string | 否 | API Key（可从环境变量读取） | - |
| baseUrl | string | 否 | API Base URL | https://api.deepseek.com/v1 |
| sessionId | string | 否 | 会话ID，提供则启用多轮对话 | - |

### 请求示例

```bash
curl -X POST http://localhost:8000/agent/code \
  -H "Content-Type: application/json" \
  -d '{
    "promptPath": "workflow/steps/agent/example_prompt.md",
    "workDir": "/home/user/project",
    "model": "deepseek-chat"
  }'
```

### 响应示例

```json
{
  "traceId": "abc123",
  "status": "success",
  "response": "AI生成的完整响应",
  "files": ["calculator.py"],
  "filesRead": ["config.py"],
  "iterations": 3
}
```

## Prompt 文件格式

Prompt 文件使用 Markdown 格式，应该清晰描述代码生成任务：

```markdown
# 任务标题

## 目标
描述要实现的功能

## 需求
1. 具体需求1
2. 具体需求2

## 代码风格
- 缩进规范
- 命名规范
- 其他要求
```

## 可用工具

AI 可以调用以下工具来操作文件系统：

### 1. read_file
读取文件内容
```json
{
  "path": "src/main.py"
}
```

### 2. write_file
写入文件（自动创建目录）
```json
{
  "path": "src/output.py",
  "content": "文件内容"
}
```

### 3. list_files
列出目录中的文件
```json
{
  "path": "src"
}
```

## 工作流程（Function Calling）

1. **API 接收请求**，获取 prompt 文件路径和会话 ID
2. **读取 prompt** 并加载会话历史（如果有）
3. **调用 LLM**，传入工具定义和消息历史
4. **AI 决策循环**：
   - AI 决定调用哪个工具（如先 `read_file` 读取现有代码）
   - 后端执行工具，获取结果
   - 将结果返回给 AI
   - AI 根据结果继续决策（如调用 `write_file` 生成代码）
   - 循环直到 AI 认为任务完成
5. **保存会话**（如果提供了 sessionId）
6. **返回结果**，包含生成的文件列表和迭代次数

### 示例执行流程

```
用户请求: "帮我重构 auth.py，添加日志"
  ↓
AI: 我需要先看看现有代码
  → 调用 read_file("auth.py")
  ↓
后端: 返回 auth.py 的内容
  ↓
AI: 我看到了代码，现在开始重构
  → 调用 write_file("auth.py", "重构后的代码")
  ↓
后端: 文件写入成功
  ↓
AI: "重构完成！我添加了日志功能..."
```

## 环境变量

可以在 `.env` 文件中配置：

```env
# LLM API 配置（必填）
API_KEY=your-api-key-here
BASE_URL=https://api.deepseek.com/v1

# Redis 配置（可选，用于会话持久化）
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0
SESSION_TTL=86400  # 会话过期时间（秒），默认 24 小时
```

## 注意事项

1. **工作目录安全**：所有文件操作都在指定的 workDir 范围内进行
2. **自动创建目录**：write_file 会自动创建所需的目录
3. **迭代限制**：最多执行 10 轮工具调用，防止无限循环
4. **会话存储**：
   - 优先使用 Redis（如果配置）实现持久化和跨进程共享
   - Redis 不可用时自动降级到内存存储
   - 会话默认保存 24 小时（可通过 SESSION_TTL 配置）
5. **API Key 配置**：通过参数或环境变量 `API_KEY` 提供
6. **模型支持**：需要支持 Function Calling 的模型（如 DeepSeek、GPT-4、Claude）

## 示例场景

### 场景1：单次代码生成
```json
{
  "promptPath": "prompts/create_api.md",
  "workDir": "/home/user/project/src"
}
```

AI 会自动：
1. 可能先 `list_files` 查看现有结构
2. 调用 `write_file` 创建 API 代码
3. 返回完成信息

### 场景2：重构现有代码
```json
{
  "promptPath": "prompts/refactor_auth.md",
  "workDir": "/home/user/project"
}
```

AI 会自动：
1. 调用 `read_file("auth.py")` 读取现有代码
2. 分析代码结构
3. 调用 `write_file("auth.py", ...)` 写入重构后的代码

### 场景3：多轮对话
```bash
# 第一次请求
curl -X POST http://localhost:8000/agent/code \
  -d '{
    "promptPath": "prompts/create_module.md",
    "sessionId": "user-123",
    "workDir": "/home/user/project"
  }'

# 第二次请求（续接上次对话）
curl -X POST http://localhost:8000/agent/code \
  -d '{
    "promptPath": "prompts/add_tests.md",
    "sessionId": "user-123",
    "workDir": "/home/user/project"
  }'
```

AI 会记住：
- 第一次创建了哪些文件
- 代码的结构和内容
- 可以基于之前的工作继续开发

## 扩展建议

1. **持久化会话**：使用 Redis 或数据库存储会话历史
2. **安全增强**：添加文件路径白名单，防止访问敏感文件
3. **工具扩展**：
   - 添加 `run_command` 工具执行编译/测试
   - 添加 `search_in_file` 工具支持代码搜索
   - 添加 `git_diff` 工具查看变更
4. **代码质量**：集成 linter、formatter 作为工具
5. **成本优化**：缓存文件内容，避免重复读取
6. **监控告警**：记录工具调用次数，设置消耗上限

## 技术细节

### Function Calling vs 传统方式

| 维度 | 传统方式 | Function Calling |
|------|---------|------------------|
| **文件操作** | 后端解析 JSON | AI 调用工具 |
| **决策能力** | 一次性生成 | 可以多次读写迭代 |
| **灵活性** | 受限于预定义格式 | AI 自主决定流程 |
| **上下文** | 需要一次性提供 | 可以按需读取 |
| **多轮对话** | 不支持 | 原生支持 |

### 会话存储架构

**Redis 模式（推荐）：**
```
Redis Key: session:{session_id}
Value: JSON 字符串（消息历史数组）
TTL: 24 小时（可配置）
```

**内存模式（Fallback）：**
```python
SESSION_STORE.memory_store = {
  "session-123": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    ...
  ]
}
```

**自动降级策略：**
1. 启动时尝试连接 Redis
2. 连接成功：所有会话存储在 Redis
3. 连接失败：自动降级到内存存储
4. 运行时 Redis 错误：单次操作降级到内存

### Redis 部署

**Docker 快速启动：**
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

**验证连接：**
```bash
redis-cli ping  # 应该返回 PONG
```

**查看会话数据：**
```bash
# 列出所有会话
redis-cli keys "session:*"

# 查看特定会话
redis-cli get "session:user-123"

# 查看会话 TTL
redis-cli ttl "session:user-123"
```
