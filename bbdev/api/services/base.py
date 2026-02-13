"""Function Calling tool base classes"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json


class Tool(ABC):
    """Tool base class"""

    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
        self.parameters = self.get_parameters()

    @abstractmethod
    def get_name(self) -> str:
        """Return tool name"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return tool description"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return tool parameter definition (JSON Schema)"""
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        """
        Execute tool

        Args:
            arguments: Tool parameters
            context: Execution context (contains logger, work_dir, etc.)

        Returns:
            Execution result (string format)
        """
        pass

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI Function Calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def safe_execute(self, arguments: Any, context: Any) -> str:
        """
        Safely execute tool (with error handling)

        Args:
            arguments: Tool parameters (can be string or dict)
            context: Execution context

        Returns:
            Execution result or error message
        """
        try:
            # Parse arguments
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments

            # Execute tool
            result = self.execute(args, context)
            return result

        except json.JSONDecodeError as e:
            error = f"Invalid JSON arguments: {str(e)}"
            if hasattr(context, "logger"):
                context.logger.error(f"Tool {self.name} - {error}")
            return json.dumps({"error": error})

        except Exception as e:
            error = f"Tool execution failed: {str(e)}"
            if hasattr(context, "logger"):
                context.logger.error(f"Tool {self.name} - {error}")
            return json.dumps({"error": error})

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolContext:
    """Tool execution context"""

    def __init__(self, work_dir: str, logger: Any = None, **kwargs):
        self.work_dir = work_dir
        self.logger = logger
        self.extra = kwargs

    def log_info(self, message: str):
        """Log info message"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")

    def log_error(self, message: str):
        """Log error message"""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"[ERROR] {message}")
