from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "dc-area-api",
    "description": "run DC synthesis and area reports",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/area")],
    "enqueues": ["dc.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    chip = require_chip(body)
    await ctx.enqueue(
        {
            "topic": "dc.verilog",
            "data": {
                "chip": chip,
                "top": body.get("top") or "DigitalTop",
                "from_area_workflow": True,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
