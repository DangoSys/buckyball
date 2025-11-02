# Agent Workflow

AI assistant workflow in BuckyBall framework, providing conversational interaction with AI models.

## API Usage

### `chat`
**Endpoint**: `POST /agent/chat`

**Function**: Conversational interaction with AI assistant

**Parameters**:
- **`message`** [Required] - Message content to send to AI
- **`model`** - AI model to use, default `"deepseek-chat"`

**Examples**:
```bash
# Basic conversation
bbdev agent --chat "--message 'Hello, can you help me with BuckyBall development?'"

# Specify model
bbdev agent --chat "--message 'Explain this Scala code' --model deepseek-chat"

# Code analysis
bbdev agent --chat "--message 'Please analyze this Chisel module and suggest optimizations'"
```

**Response**:
```json
{
  "traceId": "unique-trace-id",
  "status": "success"
}
```

## Notes

- Requires configured AI model API key
- Responses use streaming output
- Note message length limits
