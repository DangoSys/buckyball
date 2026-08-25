import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from yosys_log import req_arg

config = {
    "name": "yosys-verilog-api",
    "description": "generate verilog for yosys flow",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/verilog")],
    "enqueues": ["yosys.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {"chip": chip, "top": req_arg(body, "top") or "DigitalTop"}
    output_dir = req_arg(body, "output_dir")
    if output_dir:
        data["output_dir"] = output_dir
    log_dir = req_arg(body, "log_dir")
    if log_dir:
        data["log_dir"] = log_dir
    await ctx.enqueue({"topic": "yosys.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
