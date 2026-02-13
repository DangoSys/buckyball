"""Predefined tool sets"""

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
    """Create file operation tool set"""
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
    """Create Code Agent tool set (includes all required tools)"""
    return [
        # File operations
        ReadFileTool(),
        WriteFileTool(),
        ListFilesTool(),
        MakeDirTool(),
        DeleteFileTool(),
        GetPathTool(),
        GrepFilesTool(),
        # Agent coordination
        CallAgentTool(),
        # Workflow API
        WorkflowAPITool(),
        # Deepwiki
        DeepwikiAskTool(),
        DeepwikiReadWikiTool(),
    ]


def create_default_manager() -> ToolManager:
    """Create default tool manager"""
    manager = ToolManager()
    manager.register_tools(create_file_tools())
    return manager


def create_code_agent_manager() -> ToolManager:
    """Create Code Agent dedicated tool manager"""
    manager = ToolManager()
    manager.register_tools(create_code_agent_tools())
    return manager


# Predefined tool set configurations
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
    Get predefined tool set

    Args:
        name: Tool set name ("file_tools", "code_agent")

    Returns:
        Tool list

    Raises:
        ValueError: If tool set does not exist
    """
    config = PRESET_CONFIGS.get(name)
    if not config:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")

    return config["tools"]()


def list_presets() -> List[str]:
    """List all available predefined tool sets"""
    return list(PRESET_CONFIGS.keys())
