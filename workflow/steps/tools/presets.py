"""预定义的工具集"""

from typing import List
from .base import Tool
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
from .registry import ToolManager


def create_file_tools() -> List[Tool]:
    """创建文件操作工具集"""
    return [
        ReadFileTool(),
        WriteFileTool(),
        ListFilesTool(),
        MakeDirTool(),
        DeleteFileTool(),
        GetPathTool(),
        GrepFilesTool(),
    ]


def create_code_agent_tools() -> List[Tool]:
    """创建 Code Agent 工具集（包含所有必需工具）"""
    return [
        # 文件操作
        ReadFileTool(),
        WriteFileTool(),
        ListFilesTool(),
        MakeDirTool(),
        DeleteFileTool(),
        GetPathTool(),
        GrepFilesTool(),
        # Agent 协调
        CallAgentTool(),
        # Workflow API
        WorkflowAPITool(),
        # Deepwiki
        DeepwikiAskTool(),
        DeepwikiReadWikiTool(),
    ]


def create_default_manager() -> ToolManager:
    """创建默认工具管理器"""
    manager = ToolManager()
    manager.register_tools(create_file_tools())
    return manager


def create_code_agent_manager() -> ToolManager:
    """创建 Code Agent 专用工具管理器"""
    manager = ToolManager()
    manager.register_tools(create_code_agent_tools())
    return manager


# 预定义的工具集配置
PRESET_CONFIGS = {
    "file_tools": {
        "name": "File Operations",
        "description": "Basic file system operations",
        "tools": create_file_tools,
    },
    "code_agent": {
        "name": "Code Agent",
        "description": "Tools for code generation and manipulation",
        "tools": create_code_agent_tools,
    },
}


def get_preset(name: str) -> List[Tool]:
    """
    获取预定义工具集

    Args:
        name: 工具集名称（"file_tools", "code_agent"）

    Returns:
        工具列表

    Raises:
        ValueError: 如果工具集不存在
    """
    config = PRESET_CONFIGS.get(name)
    if not config:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")

    return config["tools"]()


def list_presets() -> List[str]:
    """列出所有可用的预定义工具集"""
    return list(PRESET_CONFIGS.keys())
