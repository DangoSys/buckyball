from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "build-compiler-api",
    "description": "build compiler",
    "flows": ["compiler"],
    "triggers": [api("POST", "/compiler/build")],
    "enqueues": ["compiler.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "compiler.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
