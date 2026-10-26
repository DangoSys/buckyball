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
    "name": "yosys-synth-api",
    "description": "run yosys synthesis for area estimation",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/synth")],
    "enqueues": ["ip.generate"],
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
        "consumer": "yosys",
        "top": req_arg(body, "top") or "DigitalTop",
        "next_topic": "yosys.synth",
    }
    output_dir = req_arg(body, "output_dir")
    if output_dir:
        data["output_dir"] = output_dir
    vcd = req_arg(body, "vcd")
    if vcd:
        data["vcd"] = vcd
    await ctx.enqueue(
        {
            "topic": "ip.generate",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
