"""Workflow 内部 API 调用工具"""

import httpx
import asyncio
from typing import Dict, Any
from .base import Tool


class WorkflowAPITool(Tool):
    """调用 Workflow 内部 API 的通用工具"""

    def get_name(self) -> str:
        return "call_workflow_api"

    def get_description(self) -> str:
        return """Call internal workflow API endpoints.
    Available endpoints:
    - /verilator/verilog: Generate Verilog
    - /verilator/build: Build verilator (params: jobs)
    - /verilator/sim: Run simulation (params: binary, batch)
    - /workload/build: Build workload (params: args)
    - /sardine/run: Run sardine tests (params: workload)"""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path (e.g., '/verilator/build')",
                },
                "params": {
                    "type": "object",
                    "description": "Request parameters as JSON object",
                    "additionalProperties": True,
                },
            },
            "required": ["endpoint"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        endpoint = arguments.get("endpoint")
        params = arguments.get("params", {})

        # 获取 workflow API 地址
        import os

        workflow_host = os.getenv("WORKFLOW_HOST", "localhost")
        workflow_port = os.getenv("WORKFLOW_PORT", "3001")
        base_url = f"http://{workflow_host}:{workflow_port}"
        url = f"{base_url}{endpoint}"

        try:
            context.log_info(f"Calling workflow API: {url}")
            context.log_info(f"Parameters: {params}")

            # 同步调用（使用 httpx 同步客户端）
            response = httpx.post(url, json=params, timeout=300.0)

            if response.status_code == 200:
                return str(response.json())
            else:
                return str(
                    {
                        "error": f"API call failed with status {response.status_code}",
                        "response": response.text[:500],
                    }
                )

        except Exception as e:
            return str({"error": f"Workflow API call failed: {str(e)}"})
