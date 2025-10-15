"""
Function Calling 工具管理模块

这个模块提供了一套完整的 Function Calling 工具管理系统，包括：
- 工具基类定义
- 工具注册和管理
- 预定义工具集
- 工具执行上下文

使用示例：

```python
from tools import create_code_agent_manager

# 创建工具管理器
manager = create_code_agent_manager()

# 获取工具定义（用于发送给 LLM）
tools_schema = manager.get_tools_schema()

# 执行工具调用
result = manager.execute_tool(
  tool_name="read_file",
  arguments={"path": "main.py"},
  work_dir="/path/to/project",
  logger=logger
)
```
"""

from .base import Tool, ToolContext
from .registry import ToolRegistry, ToolManager
from .file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from .presets import (
    create_file_tools,
    create_code_agent_tools,
    create_default_manager,
    create_code_agent_manager,
    get_preset,
    list_presets,
)

__all__ = [
    # 基础类
    "Tool",
    "ToolContext",
    # 注册和管理
    "ToolRegistry",
    "ToolManager",
    # 具体工具
    "ReadFileTool",
    "WriteFileTool",
    "ListFilesTool",
    # 预定义工具集
    "create_file_tools",
    "create_code_agent_tools",
    "create_default_manager",
    "create_code_agent_manager",
    "get_preset",
    "list_presets",
]

__version__ = "1.0.0"
