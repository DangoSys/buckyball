from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "marshal-launch-api",
    "description": "launch marshal",
    "flows": ["marshal"],
    "triggers": [api("POST", "/marshal/launch")],
    "enqueues": ["marshal.launch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "marshal.launch", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
