"""Function Calling 工具注册器"""

from typing import Dict, List, Any, Optional
from .base import Tool, ToolContext


class ToolRegistry:
    """工具注册和管理"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册一个工具"""
        self._tools[tool.name] = tool

    def register_all(self, tools: List[Tool]):
        """批量注册工具"""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """转换为 OpenAI Function Calling 格式"""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: Any, context: ToolContext) -> str:
        """
        执行工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            context: 执行上下文

        Returns:
            执行结果
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
    """工具管理器（高级封装）"""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
        self._execution_log: List[Dict[str, Any]] = []

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.registry.register(tool)

    def register_tools(self, tools: List[Tool]):
        """批量注册工具"""
        self.registry.register_all(tools)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取工具定义（OpenAI 格式）"""
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
        执行工具调用

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            work_dir: 工作目录
            logger: 日志记录器
            **kwargs: 其他上下文参数

        Returns:
            执行结果
        """
        # 创建上下文
        context = ToolContext(work_dir=work_dir, logger=logger, **kwargs)

        # 执行工具
        result = self.registry.execute(tool_name, arguments, context)

        # 记录执行日志
        self._execution_log.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result[:200] if len(result) > 200 else result,  # 截断长结果
            }
        )

        return result

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """获取执行日志"""
        return self._execution_log

    def clear_log(self):
        """清空执行日志"""
        self._execution_log.clear()

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return self.registry.list_tools()

    def __repr__(self) -> str:
        return f"ToolManager({len(self.registry)} tools registered)"
