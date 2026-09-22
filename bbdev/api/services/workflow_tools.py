"""Workflow internal API call tools"""

import httpx
from typing import Dict, Any
from .base import Tool


class WorkflowAPITool(Tool):
    """Generic tool for calling Workflow internal APIs"""

    def get_name(self) -> str:
        return "call_workflow_api"

    def get_description(self) -> str:
        return """Call internal workflow API endpoints.
    Available endpoints:
    - /verilator/verilog: Generate Verilog
    - /verilator/build: Build verilator (params: jobs)
    - /verilator/sim: Run simulation (params: binary, batch)
    - /workload/clean: Clean workload output directory
    - /workload/build: Build workload (params: chip, model, stable, rushB)"""

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

        url = f"http://localhost:3001{endpoint}"

        try:
            context.log_info(f"Calling workflow API: {url}")
            context.log_info(f"Parameters: {params}")

            # Synchronous call (using httpx sync client)
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
