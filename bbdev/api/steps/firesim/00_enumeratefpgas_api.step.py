from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "firesim-enumeratefpgas-api",
    "description": "enumerate FPGAs",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/enumeratefpgas")],
    "enqueues": ["firesim.enumeratefpgas"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    await ctx.enqueue(
        {
            "topic": "firesim.enumeratefpgas",
            "data": {"chip": chip, "_trace_id": ctx.trace_id},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
