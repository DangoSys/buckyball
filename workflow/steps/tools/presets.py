"""预定义的工具集"""

from typing import List
from .base import Tool
from .file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from .registry import ToolManager


def create_file_tools() -> List[Tool]:
    """创建文件操作工具集"""
    return [
        ReadFileTool(),
        WriteFileTool(),
        ListFilesTool(),
    ]


def create_code_agent_tools() -> List[Tool]:
    """创建 Code Agent 工具集"""
    # 目前只有文件操作工具，未来可以扩展
    return create_file_tools()


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
        "tools": create_file_tools
    },
    "code_agent": {
        "name": "Code Agent",
        "description": "Tools for code generation and manipulation",
        "tools": create_code_agent_tools
    }
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
