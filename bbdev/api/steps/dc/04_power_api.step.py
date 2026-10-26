from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "dc-power-api",
    "description": "run DC synthesis and PrimeTime PX power analysis",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/power")],
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
                **body,
                "design": design,
                "top": body.get("top"),
                "from_power_workflow": True,
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
