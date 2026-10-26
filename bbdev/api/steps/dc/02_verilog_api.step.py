from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

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
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    design = require_design(body)
    await ctx.enqueue(
        {
            "topic": "dc.verilog",
            "data": {
                "design": design,
                "top": body.get("top"),
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
