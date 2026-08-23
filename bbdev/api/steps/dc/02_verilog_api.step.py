from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
from utils.path import check_dc_rtl_args

config = {
    "name": "dc-verilog-api",
    "description": "generate verilog for dc flow",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/verilog")],
    "enqueues": ["dc.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
        subset = {"chip": chip}
        if "top" in body:
            subset["top"] = body.get("top")
        check_dc_rtl_args(subset)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {
        "chip": chip,
        "top": body.get("top") or "DigitalTop",
    }
    await ctx.enqueue({"topic": "dc.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
