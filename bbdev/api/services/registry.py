"""Function Calling tool registry"""

from typing import Dict, List, Any, Optional
from .base import Tool, ToolContext


class ToolRegistry:
    """Tool registration and management"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool"""
        self._tools[tool.name] = tool

    def register_all(self, tools: List[Tool]):
        """Batch register tools"""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get tool"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self._tools.keys())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI Function Calling format"""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: Any, context: ToolContext) -> str:
        """
        Execute tool

        Args:
            tool_name: Tool name
            arguments: Tool parameters
            context: Execution context

        Returns:
            Execution result
        """
        tool = self.get(tool_name)

        if not tool:
            return f'{{"error": "Unknown tool: {tool_name}"}}'

        return tool.safe_execute(arguments, context)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self)} tools: {', '.join(self.list_tools())})"


class ToolManager:
    """Tool manager (high-level wrapper)"""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
        self._execution_log: List[Dict[str, Any]] = []

    def register_tool(self, tool: Tool):
        """Register tool"""
        self.registry.register(tool)

    def register_tools(self, tools: List[Tool]):
        """Batch register tools"""
        self.registry.register_all(tools)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get tool definitions (OpenAI format)"""
        return self.registry.to_openai_format()

    def execute_tool(
        self,
        tool_name: str,
        arguments: Any,
        work_dir: str,
        logger: Any = None,
        **kwargs,
    ) -> str:
        """
        Execute tool call

        Args:
            tool_name: Tool name
            arguments: Tool parameters
            work_dir: Working directory
            logger: Logger
            **kwargs: Other context parameters

        Returns:
            Execution result
        """
        # Create context
        context = ToolContext(work_dir=work_dir, logger=logger, **kwargs)

        # Execute tool
        result = self.registry.execute(tool_name, arguments, context)

        # Log execution
        self._execution_log.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                # Truncate long results
                "result": result[:200] if len(result) > 200 else result,
            }
        )

        return result

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log"""
        return self._execution_log

    def clear_log(self):
        """Clear execution log"""
        self._execution_log.clear()

    def get_tool_names(self) -> List[str]:
        """Get all tool names"""
        return self.registry.list_tools()

    def __repr__(self) -> str:
        return f"ToolManager({len(self.registry)} tools registered)"
