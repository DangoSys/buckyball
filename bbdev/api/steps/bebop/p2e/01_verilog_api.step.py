from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path

config = {
    "name": "bebop-p2e-verilog-api",
    "description": "Generate SystemVerilog for P2E DDR4 backdoor",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/verilog")],
    "enqueues": ["bebop.p2e.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    data = {"chip": chip}
    if body.get("output_dir"):
        data["output_dir"] = body["output_dir"]
    await ctx.enqueue({"topic": "bebop.p2e.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
