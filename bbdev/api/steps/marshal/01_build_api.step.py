from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "marshal-build-api",
    "description": "build marshal",
    "flows": ["marshal"],
    "triggers": [api("POST", "/marshal/build")],
    "enqueues": ["marshal.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "marshal.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
