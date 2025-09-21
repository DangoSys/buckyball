# Agent 工作流

BuckyBall 框架中的 AI 助手工作流，提供与 AI 模型的对话交互功能。

## API 使用说明

### `chat`
**端点**: `POST /agent/chat`

**功能**: 与 AI 助手进行对话交互

**参数**:
- **`message`** [必选] - 发送给 AI 的消息内容
- **`model`** - 使用的 AI 模型，默认 `"deepseek-chat"`

**示例**:
```bash
# 基本对话
bbdev agent --chat "--message 'Hello, can you help me with BuckyBall development?'"

# 指定模型
bbdev agent --chat "--message 'Explain this Scala code' --model deepseek-chat"

# 代码分析
bbdev agent --chat "--message 'Please analyze this Chisel module and suggest optimizations'"
```

**响应**:
```json
{
  "traceId": "unique-trace-id",
  "status": "success"
}
```

## 注意事项

- 需要配置 AI 模型的 API 密钥
- 响应采用流式输出
- 注意消息长度限制
