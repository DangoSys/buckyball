# Function Calling Tool Management System

A modular and extensible Function Calling tool management framework for AI Agent interaction with external systems.

## 📁 Directory Structure

```
tools/
├── __init__.py       # Module exports
├── base.py          # Tool base class and context
├── registry.py      # Tool registry and manager
├── file_tools.py    # File operation tools
├── presets.py       # Predefined tool sets
└── README.md        # This document
```

## 🚀 Quick Start

### 1. Using Predefined Tool Manager

```python
from steps.tools import create_code_agent_manager

# Create tool manager (file_tools already registered)
manager = create_code_agent_manager()

# Get tool definitions (send to LLM)
tools_schema = manager.get_tools_schema()

# Call LLM
response = llm.chat(
  messages=messages,
  tools=tools_schema  # Pass in tool definitions
)

# Execute tool calls returned by LLM
if response.tool_calls:
  for tool_call in response.tool_calls:
    result = manager.execute_tool(
      tool_name=tool_call.function.name,
      arguments=tool_call.function.arguments,
      work_dir="/path/to/project",
      logger=logger
    )
```

### 2. Custom Tools

```python
from steps.tools import Tool, ToolManager
import json

class RunCommandTool(Tool):
  """Command execution tool"""

  def get_name(self) -> str:
    return "run_command"

  def get_description(self) -> str:
    return "Execute a shell command"

  def get_parameters(self) -> dict:
    return {
      "type": "object",
      "properties": {
        "command": {"type": "string", "description": "Command to execute"}
      },
      "required": ["command"]
    }

  def execute(self, arguments: dict, context) -> str:
    import subprocess
    cmd = arguments.get("command")

    try:
      result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
      )

      return json.dumps({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
      })
    except Exception as e:
      return json.dumps({"error": str(e)})

# Register custom tool
manager = ToolManager()
manager.register_tool(RunCommandTool())
```

## 📚 Core Concepts

### Tool (Base Class)

All tools inherit from the `Tool` base class:

```python
class MyTool(Tool):
  def get_name(self) -> str:
    """Tool name (unique identifier)"""
    return "my_tool"

  def get_description(self) -> str:
    """Tool description (tells AI what this tool does)"""
    return "My custom tool"

  def get_parameters(self) -> dict:
    """Parameter definition (JSON Schema format)"""
    return {
      "type": "object",
      "properties": {
        "param1": {"type": "string", "description": "Parameter 1"},
        "param2": {"type": "integer", "description": "Parameter 2"}
      },
      "required": ["param1"]
    }

  def execute(self, arguments: dict, context) -> str:
    """
    Execute tool logic

    Args:
      arguments: Parameters passed by AI
      context: ToolContext object (contains work_dir, logger, etc.)

    Returns:
      Execution result (string, can be JSON)
    """
    # Implement tool logic
    return json.dumps({"result": "success"})
```

### ToolContext (Execution Context)

Execution environment provided to tools:

```python
context = ToolContext(
  work_dir="/path/to/project",  # Working directory
  logger=logger,                 # Logger
  extra_key="extra_value"        # Custom extension fields
)

# Use in tool
class MyTool(Tool):
  def execute(self, arguments, context):
    context.log_info("Starting execution")
    work_dir = context.work_dir
    custom = context.extra.get("extra_key")
    # ...
```

### ToolRegistry (Tool Registry)

Manage tool registration and lookup:

```python
from steps.tools import ToolRegistry, ReadFileTool, WriteFileTool

registry = ToolRegistry()
registry.register(ReadFileTool())
registry.register(WriteFileTool())

# Get tool
tool = registry.get("read_file")

# List all tools
tools = registry.list_tools()  # ['read_file', 'write_file']

# Convert to OpenAI format
schema = registry.to_openai_format()
```

### ToolManager (Tool Manager)

High-level wrapper providing more convenient interface:

```python
from steps.tools import ToolManager

manager = ToolManager()
manager.register_tools([ReadFileTool(), WriteFileTool()])

# Get tool definitions
schema = manager.get_tools_schema()

# Execute tool
result = manager.execute_tool(
  tool_name="read_file",
  arguments={"path": "main.py"},
  work_dir="/project"
)

# View execution log
log = manager.get_execution_log()
```

## 🛠️ Built-in Tools

### File Operation Tools

#### read_file
Read file content

```json
{
  "name": "read_file",
  "parameters": {
    "path": "relative/path/to/file.txt"
  }
}
```

#### write_file
Write file content (automatically creates directories)

```json
{
  "name": "write_file",
  "parameters": {
    "path": "output/file.txt",
    "content": "file content here"
  }
}
```

#### list_files
List files in directory

```json
{
  "name": "list_files",
  "parameters": {
    "path": "src"  // Optional, defaults to current directory
  }
}
```

## 🎨 Predefined Tool Sets

```python
from steps.tools import get_preset, list_presets

# View available tool sets
presets = list_presets()  # ['file_tools', 'code_agent']

# Get tool set
tools = get_preset("file_tools")  # Returns [ReadFileTool, WriteFileTool, ...]

# Create manager
from steps.tools import create_code_agent_manager
manager = create_code_agent_manager()
```

## 📝 Complete Examples

### Agent Integration Example

```python
from steps.tools import create_code_agent_manager
import httpx
import json

async def run_agent(prompt: str, work_dir: str):
  # 1. Create tool manager
  manager = create_code_agent_manager()
  tools_schema = manager.get_tools_schema()

  # 2. Initialize conversation
  messages = [
    {"role": "system", "content": "You are a code assistant"},
    {"role": "user", "content": prompt}
  ]

  # 3. AI loop
  max_iterations = 10
  for i in range(max_iterations):
    # Call LLM
    response = await call_llm(messages, tools_schema)
    assistant_msg = response["choices"][0]["message"]
    messages.append(assistant_msg)

    # Check for tool calls
    if not assistant_msg.get("tool_calls"):
      print(f"Done! Final response: {assistant_msg['content']}")
      break

    # Execute all tool calls
    for tool_call in assistant_msg["tool_calls"]:
      result = manager.execute_tool(
        tool_name=tool_call["function"]["name"],
        arguments=tool_call["function"]["arguments"],
        work_dir=work_dir
      )

      # Add tool result to conversation
      messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": result
      })
```

## 🔒 Security Features

### Path Security

All file operation tools include path traversal checks:

```python
# ✅ Allowed
read_file("src/main.py")

# ❌ Denied (path traversal)
read_file("../../../etc/passwd")
```

### Error Handling

Tool execution automatically catches exceptions:

```python
# Internal exceptions are caught and returned as error messages
result = manager.execute_tool("read_file", {"path": "nonexist.txt"})
# Returns: {"error": "File not found: nonexist.txt"}
```

## 🧪 Testing

Create test file:

```python
# test_tools.py
from steps.tools import create_code_agent_manager
import tempfile
import os

def test_file_tools():
  manager = create_code_agent_manager()

  with tempfile.TemporaryDirectory() as tmpdir:
    # Test write_file
    result = manager.execute_tool(
      "write_file",
      {"path": "test.txt", "content": "hello"},
      work_dir=tmpdir
    )
    assert "success" in result

    # Test read_file
    result = manager.execute_tool(
      "read_file",
      {"path": "test.txt"},
      work_dir=tmpdir
    )
    assert result == "hello"

    print("✅ Test passed")

if __name__ == "__main__":
  test_file_tools()
```

## 📦 Extending Tools

### Adding New Tool Categories

Create new file `network_tools.py`:

```python
from .base import Tool
import json
import httpx

class HttpGetTool(Tool):
  def get_name(self) -> str:
    return "http_get"

  def get_description(self) -> str:
    return "Make HTTP GET request"

  def get_parameters(self) -> dict:
    return {
      "type": "object",
      "properties": {
        "url": {"type": "string"}
      },
      "required": ["url"]
    }

  def execute(self, arguments, context):
    url = arguments["url"]
    response = httpx.get(url)
    return json.dumps({
      "status": response.status_code,
      "body": response.text[:1000]
    })
```

Then add to `presets.py`:

```python
def create_network_tools():
  from .network_tools import HttpGetTool
  return [HttpGetTool()]
```

## 🎯 Best Practices

1. **Tool Naming**: Use clear verb+noun format (`read_file`, `list_users`)
2. **Parameter Descriptions**: Describe each parameter in detail to help AI use them correctly
3. **Error Handling**: Return JSON format error messages with `error` field
4. **Logging**: Use `context.log_info/log_error` to record key operations
5. **Security Checks**: Validate input parameters, prevent path traversal and other security issues
6. **Result Format**: Return JSON strings or plain text, maintain consistency

## 🤝 Contributing

Steps to add new tools:

1. Inherit from `Tool` base class
2. Implement 4 abstract methods
3. Add to appropriate tool set in `presets.py`
4. Export in `__init__.py`
5. Update README documentation

## 📄 License

Same as main project
