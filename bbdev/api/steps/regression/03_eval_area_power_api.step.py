from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "regression-eval-area-power-api",
    "description": "DC area+freq for regression",
    "flows": ["regression"],
    "triggers": [api("POST", "/regression/eval-area-power")],
    "enqueues": ["regression.eval-area-power"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    chip = require_chip(body)
    data = {
        "chip": chip,
        "_trace_id": ctx.trace_id,
    }
    top = body.get("top")
    if isinstance(top, str) and top:
        data["top"] = top
    await ctx.enqueue({"topic": "regression.eval-area-power", "data": data})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
