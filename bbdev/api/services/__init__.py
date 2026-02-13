"""
Function Calling Tool Management Module

This module provides a complete Function Calling tool management system, including:
- Tool base class definitions
- Tool registration and management
- Predefined tool sets
- Tool execution context

Usage example:

```python
from tools import create_code_agent_manager

# Create tool manager
manager = create_code_agent_manager()

# Get tool definitions (to send to LLM)
tools_schema = manager.get_tools_schema()

# Execute tool call
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
from .file_tools import (
    ReadFileTool,
    WriteFileTool,
    ListFilesTool,
    MakeDirTool,
    DeleteFileTool,
    GetPathTool,
    GrepFilesTool,
)
from .workflow_tools import WorkflowAPITool
from .deepwiki_tools import DeepwikiAskTool, DeepwikiReadWikiTool
from .agent_tools import CallAgentTool
from .presets import (
    create_file_tools,
    create_code_agent_tools,
    create_default_manager,
    create_code_agent_manager,
    get_preset,
    list_presets,
)

__all__ = [
    # Base classes
    "Tool",
    "ToolContext",
    # Registration and management
    "ToolRegistry",
    "ToolManager",
    # File operation tools
    "ReadFileTool",
    "WriteFileTool",
    "ListFilesTool",
    "MakeDirTool",
    "DeleteFileTool",
    "GetPathTool",
    "GrepFilesTool",
    # Agent coordination
    "CallAgentTool",
    # API tools
    "WorkflowAPITool",
    "DeepwikiAskTool",
    "DeepwikiReadWikiTool",
    # Predefined tool sets
    "create_file_tools",
    "create_code_agent_tools",
    "create_default_manager",
    "create_code_agent_manager",
    "get_preset",
    "list_presets",
]

__version__ = "1.0.0"
