from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "firesim-buildbitstream-api",
    "description": "build bitstream",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/buildbitstream")],
    "enqueues": ["firesim.buildbitstream"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    await ctx.enqueue(
        {
            "topic": "firesim.buildbitstream",
            "data": {"chip": chip, "_trace_id": ctx.trace_id},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
