from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "firesim-buildbitstream-api",
    "description": "build bitstream",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/buildbitstream")],
    "enqueues": ["firesim.buildbitstream"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "firesim.buildbitstream", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
