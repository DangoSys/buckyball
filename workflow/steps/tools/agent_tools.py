"""Agent 协调工具"""

import httpx
import uuid
from typing import Dict, Any
from .base import Tool


class CallAgentTool(Tool):
    """调用其他 Agent 的工具"""

    def get_name(self) -> str:
        return "call_agent"

    def get_description(self) -> str:
        return """Call another agent (spec_agent, code_agent, review_agent, verify_agent) with a task.
    Use this to delegate work to specialized agents.
    Returns the agent's response and generated files."""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_role": {
                    "type": "string",
                    "enum": ["spec", "code", "review", "verify"],
                    "description": "Which agent to call (spec/code/review/verify)",
                },
                "task_description": {
                    "type": "string",
                    "description": "Task description or instructions for the agent",
                },
                "context_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: List of file paths the agent should read for context",
                },
                "model": {
                    "type": "string",
                    "description": "Optional: LLM model to use (will inherit from parent if not specified)",
                },
            },
            "required": ["agent_role", "task_description"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        agent_role = arguments.get("agent_role")
        task_description = arguments.get("task_description")
        context_files = arguments.get("context_files", [])
        # 优先使用参数中的 model，否则从 context 继承
        model = arguments.get("model") or context.extra.get("model")

        # 生成临时任务文件
        import os
        import tempfile

        # 创建临时任务文件
        task_content = task_description

        # 如果指定了上下文文件，添加引用
        if context_files:
            task_content += "\n\n## Context Files\n"
            for filepath in context_files:
                task_content += f"- {filepath}\n"

        # 写入临时文件
        temp_dir = os.path.join(context.work_dir, ".agent_tasks")
        os.makedirs(temp_dir, exist_ok=True)

        task_id = str(uuid.uuid4())[:8]
        task_file = os.path.join(temp_dir, f"task_{agent_role}_{task_id}.md")

        with open(task_file, "w", encoding="utf-8") as f:
            f.write(task_content)

        context.log_info(f"Created task file: {task_file}")
        context.log_info(f"Calling {agent_role}_agent with task")

        try:
            # 获取 workflow API 地址（从环境变量或使用默认值）
            import os

            workflow_host = os.getenv("WORKFLOW_HOST", "localhost")
            workflow_port = os.getenv("WORKFLOW_PORT", "3001")
            base_url = f"http://{workflow_host}:{workflow_port}"
            url = f"{base_url}/agent"

            context.log_info(f"Calling workflow API at: {url}")

            payload = {
                "agentRole": agent_role,
                "promptPath": task_file,
                "workDir": context.work_dir,
            }

            # 如果指定了 model，添加到 payload
            if model:
                payload["model"] = model
                context.log_info(f"Using model: {model}")

            response = httpx.post(url, json=payload, timeout=600.0)

            if response.status_code == 200:
                result = response.json()

                # 清理临时文件
                try:
                    os.remove(task_file)
                except Exception:
                    pass

                return str(
                    {
                        "status": "success",
                        "agent": agent_role,
                        "result": result,
                        "files": result.get("files", []),
                    }
                )
            else:
                return str(
                    {
                        "error": f"Agent call failed with status {response.status_code}",
                        "response": response.text[:500],
                    }
                )

        except Exception as e:
            return str({"error": f"Failed to call agent: {str(e)}"})
