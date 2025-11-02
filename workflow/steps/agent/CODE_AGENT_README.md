# Code Agent Step Usage Guide

## Overview
Code Agent Step is an AI-based intelligent code assistant that uses **Function Calling** technology to let AI autonomously decide how to read and generate code.

### ✨ Core Features
- 🤖 **Intelligent Decision Making**: AI decides which files to read first, then how to generate code
- 🔄 **Multi-turn Conversation**: Supports session management, AI can remember previous operations
- 🛠️ **Tool Calling**: AI can use tools like read_file, write_file, list_files
- 📝 **Auto Iteration**: AI can read and write files multiple times until task completion

## File Structure
- `02_code_agent_api_step.py` - API entry point, receives requests
- `02_code_agent_event_step.py` - Event handler, executes AI calls and file operations
- `example_prompt.md` - Example prompt file

## API Interface

### Endpoint
```
POST /agent/code
```

### Request Parameters

| Parameter | Type | Required | Description | Default |
|------|------|------|------|--------|
| promptPath | string | Yes | Markdown prompt file path | - |
| workDir | string | No | Working directory path | Current directory |
| model | string | No | AI model name | deepseek-chat |
| apiKey | string | No | API Key (can read from env variables) | - |
| baseUrl | string | No | API Base URL | https://api.deepseek.com/v1 |
| sessionId | string | No | Session ID, enables multi-turn conversation if provided | - |

### Request Example

```bash
curl -X POST http://localhost:8000/agent/code \
  -H "Content-Type: application/json" \
  -d '{
    "promptPath": "workflow/steps/agent/example_prompt.md",
    "workDir": "/home/user/project",
    "model": "deepseek-chat"
  }'
```

### Response Example

```json
{
  "traceId": "abc123",
  "status": "success",
  "response": "Complete AI-generated response",
  "files": ["calculator.py"],
  "filesRead": ["config.py"],
  "iterations": 3
}
```

## Prompt File Format

Prompt files use Markdown format and should clearly describe the code generation task:

```markdown
# Task Title

## Objective
Describe the functionality to implement

## Requirements
1. Specific requirement 1
2. Specific requirement 2

## Code Style
- Indentation rules
- Naming conventions
- Other requirements
```

## Available Tools

AI can call the following tools to manipulate the file system:

### 1. read_file
Read file content
```json
{
  "path": "src/main.py"
}
```

### 2. write_file
Write file (automatically creates directories)
```json
{
  "path": "src/output.py",
  "content": "file content"
}
```

### 3. list_files
List files in directory
```json
{
  "path": "src"
}
```

## Workflow (Function Calling)

1. **API receives request**, gets prompt file path and session ID
2. **Read prompt** and load session history (if any)
3. **Call LLM**, pass in tool definitions and message history
4. **AI decision loop**:
   - AI decides which tool to call (e.g., `read_file` to read existing code first)
   - Backend executes tool, gets result
   - Return result to AI
   - AI continues decision making based on result (e.g., call `write_file` to generate code)
   - Loop until AI considers task complete
5. **Save session** (if sessionId provided)
6. **Return result**, including list of generated files and iteration count

### Example Execution Flow

```
User request: "Help me refactor auth.py, add logging"
  ↓
AI: I need to see the existing code first
  → Call read_file("auth.py")
  ↓
Backend: Return auth.py content
  ↓
AI: I see the code, now starting refactoring
  → Call write_file("auth.py", "refactored code")
  ↓
Backend: File write successful
  ↓
AI: "Refactoring complete! I added logging functionality..."
```

## Environment Variables

Can be configured in `.env` file:

```env
# LLM API configuration (required)
API_KEY=your-api-key-here
BASE_URL=https://api.deepseek.com/v1

# Redis configuration (optional, for session persistence)
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0
SESSION_TTL=86400  # Session expiration time (seconds), default 24 hours
```

## Notes

1. **Working Directory Security**: All file operations are within the specified workDir scope
2. **Auto Create Directories**: write_file automatically creates required directories
3. **Iteration Limit**: Maximum 10 rounds of tool calls to prevent infinite loops
4. **Session Storage**:
   - Prefer Redis (if configured) for persistence and cross-process sharing
   - Automatically downgrade to memory storage when Redis unavailable
   - Sessions saved for 24 hours by default (configurable via SESSION_TTL)
5. **API Key Configuration**: Provided via parameter or `API_KEY` environment variable
6. **Model Support**: Requires models that support Function Calling (e.g., DeepSeek, GPT-4, Claude)

## Example Scenarios

### Scenario 1: Single Code Generation
```json
{
  "promptPath": "prompts/create_api.md",
  "workDir": "/home/user/project/src"
}
```

AI will automatically:
1. Possibly call `list_files` first to view existing structure
2. Call `write_file` to create API code
3. Return completion information

### Scenario 2: Refactor Existing Code
```json
{
  "promptPath": "prompts/refactor_auth.md",
  "workDir": "/home/user/project"
}
```

AI will automatically:
1. Call `read_file("auth.py")` to read existing code
2. Analyze code structure
3. Call `write_file("auth.py", ...)` to write refactored code

### Scenario 3: Multi-turn Conversation
```bash
# First request
curl -X POST http://localhost:8000/agent/code \
  -d '{
    "promptPath": "prompts/create_module.md",
    "sessionId": "user-123",
    "workDir": "/home/user/project"
  }'

# Second request (continue from previous conversation)
curl -X POST http://localhost:8000/agent/code \
  -d '{
    "promptPath": "prompts/add_tests.md",
    "sessionId": "user-123",
    "workDir": "/home/user/project"
  }'
```

AI will remember:
- Which files were created in the first round
- Code structure and content
- Can continue development based on previous work

## Extension Suggestions

1. **Session Persistence**: Use Redis or database to store session history
2. **Security Enhancement**: Add file path whitelist to prevent access to sensitive files
3. **Tool Extensions**:
   - Add `run_command` tool to execute compilation/testing
   - Add `search_in_file` tool for code search support
   - Add `git_diff` tool to view changes
4. **Code Quality**: Integrate linter, formatter as tools
5. **Cost Optimization**: Cache file content to avoid repeated reads
6. **Monitoring and Alerting**: Log tool call counts, set consumption limits

## Technical Details

### Function Calling vs Traditional Approach

| Dimension | Traditional Approach | Function Calling |
|------|---------|------------------|
| **File Operations** | Backend parses JSON | AI calls tools |
| **Decision Making** | One-time generation | Can iterate with multiple reads/writes |
| **Flexibility** | Limited by predefined format | AI autonomously decides workflow |
| **Context** | Must provide all at once | Can read on demand |
| **Multi-turn Conversation** | Not supported | Natively supported |

### Session Storage Architecture

**Redis Mode (Recommended):**
```
Redis Key: session:{session_id}
Value: JSON string (message history array)
TTL: 24 hours (configurable)
```

**Memory Mode (Fallback):**
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

**Auto-downgrade Strategy:**
1. Attempt to connect to Redis at startup
2. Connection successful: Store all sessions in Redis
3. Connection failed: Automatically downgrade to memory storage
4. Runtime Redis error: Single operation downgrades to memory

### Redis Deployment

**Docker Quick Start:**
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

**Verify Connection:**
```bash
redis-cli ping  # Should return PONG
```

**View Session Data:**
```bash
# List all sessions
redis-cli keys "session:*"

# View specific session
redis-cli get "session:user-123"

# View session TTL
redis-cli ttl "session:user-123"
```
