import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

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
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {
        "chip": chip,
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
    log_dir = req_arg(body, "log_dir")
    if log_dir:
        data["log_dir"] = log_dir
    await ctx.enqueue(
        {
            "topic": "ip.generate",
            "data": {**data, "_trace_id": ctx.trace_id},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
