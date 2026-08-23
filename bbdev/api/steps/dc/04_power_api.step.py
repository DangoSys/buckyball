from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
from utils.path import check_dc_power_args

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
        chip = require_chip(body)
        check_dc_power_args({k: v for k, v in body.items() if k != "config"})
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})
    await ctx.enqueue(
        {
            "topic": "dc.verilog",
            "data": {
                **body,
                "chip": chip,
                "top": body.get("top") or "DigitalTop",
                "from_power_workflow": True,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
