# Function Calling 工具管理系统

一个模块化、可扩展的 Function Calling 工具管理框架，用于 AI Agent 与外部系统交互。

## 📁 目录结构

```
tools/
├── __init__.py       # 模块导出
├── base.py          # 工具基类和上下文
├── registry.py      # 工具注册器和管理器
├── file_tools.py    # 文件操作工具
├── presets.py       # 预定义工具集
└── README.md        # 本文档
```

## 🚀 快速开始

### 1. 使用预定义工具管理器

```python
from steps.tools import create_code_agent_manager

# 创建工具管理器（已注册 file_tools）
manager = create_code_agent_manager()

# 获取工具定义（发送给 LLM）
tools_schema = manager.get_tools_schema()

# 调用 LLM
response = llm.chat(
  messages=messages,
  tools=tools_schema  # 传入工具定义
)

# 执行 LLM 返回的工具调用
if response.tool_calls:
  for tool_call in response.tool_calls:
    result = manager.execute_tool(
      tool_name=tool_call.function.name,
      arguments=tool_call.function.arguments,
      work_dir="/path/to/project",
      logger=logger
    )
```

### 2. 自定义工具

```python
from steps.tools import Tool, ToolManager
import json

class RunCommandTool(Tool):
  """执行命令工具"""

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

# 注册自定义工具
manager = ToolManager()
manager.register_tool(RunCommandTool())
```

## 📚 核心概念

### Tool（工具基类）

所有工具都继承自 `Tool` 基类：

```python
class MyTool(Tool):
  def get_name(self) -> str:
    """工具名称（唯一标识）"""
    return "my_tool"

  def get_description(self) -> str:
    """工具描述（告诉 AI 这个工具做什么）"""
    return "My custom tool"

  def get_parameters(self) -> dict:
    """参数定义（JSON Schema 格式）"""
    return {
      "type": "object",
      "properties": {
        "param1": {"type": "string", "description": "参数1"},
        "param2": {"type": "integer", "description": "参数2"}
      },
      "required": ["param1"]
    }

  def execute(self, arguments: dict, context) -> str:
    """
    执行工具逻辑

    Args:
      arguments: AI 传入的参数
      context: ToolContext 对象（包含 work_dir, logger 等）

    Returns:
      执行结果（字符串，可以是 JSON）
    """
    # 实现工具逻辑
    return json.dumps({"result": "success"})
```

### ToolContext（执行上下文）

提供给工具的执行环境：

```python
context = ToolContext(
  work_dir="/path/to/project",  # 工作目录
  logger=logger,                 # 日志记录器
  extra_key="extra_value"        # 自定义扩展字段
)

# 在工具中使用
class MyTool(Tool):
  def execute(self, arguments, context):
    context.log_info("开始执行")
    work_dir = context.work_dir
    custom = context.extra.get("extra_key")
    # ...
```

### ToolRegistry（工具注册器）

管理工具的注册和查找：

```python
from steps.tools import ToolRegistry, ReadFileTool, WriteFileTool

registry = ToolRegistry()
registry.register(ReadFileTool())
registry.register(WriteFileTool())

# 获取工具
tool = registry.get("read_file")

# 列出所有工具
tools = registry.list_tools()  # ['read_file', 'write_file']

# 转换为 OpenAI 格式
schema = registry.to_openai_format()
```

### ToolManager（工具管理器）

高级封装，提供更便捷的接口：

```python
from steps.tools import ToolManager

manager = ToolManager()
manager.register_tools([ReadFileTool(), WriteFileTool()])

# 获取工具定义
schema = manager.get_tools_schema()

# 执行工具
result = manager.execute_tool(
  tool_name="read_file",
  arguments={"path": "main.py"},
  work_dir="/project"
)

# 查看执行日志
log = manager.get_execution_log()
```

## 🛠️ 内置工具

### 文件操作工具

#### read_file
读取文件内容

```json
{
  "name": "read_file",
  "parameters": {
    "path": "relative/path/to/file.txt"
  }
}
```

#### write_file
写入文件内容（自动创建目录）

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
列出目录中的文件

```json
{
  "name": "list_files",
  "parameters": {
    "path": "src"  // 可选，默认当前目录
  }
}
```

## 🎨 预定义工具集

```python
from steps.tools import get_preset, list_presets

# 查看可用的工具集
presets = list_presets()  # ['file_tools', 'code_agent']

# 获取工具集
tools = get_preset("file_tools")  # 返回 [ReadFileTool, WriteFileTool, ...]

# 创建管理器
from steps.tools import create_code_agent_manager
manager = create_code_agent_manager()
```

## 📝 完整示例

### Agent 集成示例

```python
from steps.tools import create_code_agent_manager
import httpx
import json

async def run_agent(prompt: str, work_dir: str):
  # 1. 创建工具管理器
  manager = create_code_agent_manager()
  tools_schema = manager.get_tools_schema()

  # 2. 初始化对话
  messages = [
    {"role": "system", "content": "You are a code assistant"},
    {"role": "user", "content": prompt}
  ]

  # 3. AI 循环
  max_iterations = 10
  for i in range(max_iterations):
    # 调用 LLM
    response = await call_llm(messages, tools_schema)
    assistant_msg = response["choices"][0]["message"]
    messages.append(assistant_msg)

    # 检查是否有工具调用
    if not assistant_msg.get("tool_calls"):
      print(f"完成！最终回复: {assistant_msg['content']}")
      break

    # 执行所有工具调用
    for tool_call in assistant_msg["tool_calls"]:
      result = manager.execute_tool(
        tool_name=tool_call["function"]["name"],
        arguments=tool_call["function"]["arguments"],
        work_dir=work_dir
      )

      # 添加工具结果到对话
      messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": result
      })
```

## 🔒 安全特性

### 路径安全

所有文件操作工具都包含路径穿越检查：

```python
# ✅ 允许
read_file("src/main.py")

# ❌ 拒绝（路径穿越）
read_file("../../../etc/passwd")
```

### 错误处理

工具执行自动捕获异常：

```python
# 工具内部异常会被捕获并返回错误信息
result = manager.execute_tool("read_file", {"path": "nonexist.txt"})
# 返回: {"error": "File not found: nonexist.txt"}
```

## 🧪 测试

创建测试文件：

```python
# test_tools.py
from steps.tools import create_code_agent_manager
import tempfile
import os

def test_file_tools():
  manager = create_code_agent_manager()

  with tempfile.TemporaryDirectory() as tmpdir:
    # 测试 write_file
    result = manager.execute_tool(
      "write_file",
      {"path": "test.txt", "content": "hello"},
      work_dir=tmpdir
    )
    assert "success" in result

    # 测试 read_file
    result = manager.execute_tool(
      "read_file",
      {"path": "test.txt"},
      work_dir=tmpdir
    )
    assert result == "hello"

    print("✅ 测试通过")

if __name__ == "__main__":
  test_file_tools()
```

## 📦 扩展工具

### 添加新工具类别

创建新文件 `network_tools.py`：

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

然后在 `presets.py` 中添加：

```python
def create_network_tools():
  from .network_tools import HttpGetTool
  return [HttpGetTool()]
```

## 🎯 最佳实践

1. **工具命名**：使用清晰的动词+名词格式（`read_file`, `list_users`）
2. **参数描述**：详细描述每个参数的作用，帮助 AI 正确使用
3. **错误处理**：返回 JSON 格式的错误信息，包含 `error` 字段
4. **日志记录**：使用 `context.log_info/log_error` 记录关键操作
5. **安全检查**：验证输入参数，防止路径穿越等安全问题
6. **结果格式**：返回 JSON 字符串或纯文本，保持一致性

## 🤝 贡献

添加新工具步骤：

1. 继承 `Tool` 基类
2. 实现 4 个抽象方法
3. 在 `presets.py` 中添加到相应工具集
4. 在 `__init__.py` 中导出
5. 更新 README 文档

## 📄 许可

与主项目相同
