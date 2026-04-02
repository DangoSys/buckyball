from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "Pegasus Buildbitstream",
    "description": "build pegasus bitstream",
    "flows": ["pegasus"],
    "triggers": [api("POST", "/pegasus/buildbitstream")],
    "enqueues": ["pegasus.buildbitstream"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "pegasus.buildbitstream", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
