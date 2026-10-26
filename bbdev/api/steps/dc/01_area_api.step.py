from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "dc-area-api",
    "description": "run DC synthesis and area reports",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/area")],
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
                "top": body.get("top") or "DigitalTop",
                "from_area_workflow": True,
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
