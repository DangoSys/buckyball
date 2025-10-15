"""Function Calling 工具基础类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json


class Tool(ABC):
    """工具基类"""

    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
        self.parameters = self.get_parameters()

    @abstractmethod
    def get_name(self) -> str:
        """返回工具名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """返回工具描述"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """返回工具参数定义（JSON Schema）"""
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        """
        执行工具

        Args:
            arguments: 工具参数
            context: 执行上下文（包含 logger、work_dir 等）

        Returns:
            执行结果（字符串格式）
        """
        pass

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def safe_execute(self, arguments: Any, context: Any) -> str:
        """
        安全执行工具（带错误处理）

        Args:
            arguments: 工具参数（可能是字符串或字典）
            context: 执行上下文

        Returns:
            执行结果或错误信息
        """
        try:
            # 解析参数
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments

            # 执行工具
            result = self.execute(args, context)
            return result

        except json.JSONDecodeError as e:
            error = f"Invalid JSON arguments: {str(e)}"
            if hasattr(context, 'logger'):
                context.logger.error(f"Tool {self.name} - {error}")
            return json.dumps({"error": error})

        except Exception as e:
            error = f"Tool execution failed: {str(e)}"
            if hasattr(context, 'logger'):
                context.logger.error(f"Tool {self.name} - {error}")
            return json.dumps({"error": error})

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolContext:
    """工具执行上下文"""

    def __init__(self, work_dir: str, logger: Any = None, **kwargs):
        self.work_dir = work_dir
        self.logger = logger
        self.extra = kwargs

    def log_info(self, message: str):
        """记录信息日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")

    def log_error(self, message: str):
        """记录错误日志"""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"[ERROR] {message}")
