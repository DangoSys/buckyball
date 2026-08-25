from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir

config = {
    "name": "verilator-build-api",
    "description": "build verilator executable",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/build")],
    "enqueues": ["verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    data = {
        "chip": chip,
        "jobs": body.get("jobs", 16),
        "output_dir": rtl_dir(bbdir, chip, "verilog", body.get("output_dir")),
    }
    await ctx.enqueue({"topic": "verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
