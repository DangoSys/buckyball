from utils.event_common import require_target
import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from yosys_log import req_arg

config = {
    "name": "yosys-run",
    "description": "run yosys flow",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/run")],
    "enqueues": ["yosys.run"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    unknown = body.keys() - {
        "device",
        "core",
        "design",
        "top",
        "output-dir",
        "output_dir",
        "vcd",
    }
    if unknown:
        return ApiResponse(
            status=400,
            body={"error": f"Unknown parameter(s): {', '.join(sorted(unknown))}"},
        )
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {
        "design": design,
        "top": req_arg(body, "top"),
        "vcd": req_arg(body, "vcd"),
        "from_run_workflow": True,
    }
    output_dir = req_arg(body, "output_dir")
    if output_dir:
        data["output_dir"] = output_dir
    await ctx.enqueue(
        {
            "topic": "yosys.run",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
