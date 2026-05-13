from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-bemu-build-api",
    "description": "Build bebop bemu binary",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/build")],
    "enqueues": ["bebop.bemu.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "bebop.bemu.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
