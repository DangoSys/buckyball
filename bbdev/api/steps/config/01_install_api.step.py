from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "config-install-api",
    "description": "install all chip configs under examples/chips",
    "flows": ["config"],
    "triggers": [api("POST", "/config/install")],
    "enqueues": ["config.install"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if body:
        unknown = sorted(body.keys())
        return ApiResponse(
            status=400,
            body={"error": f"config install takes no parameters: {', '.join(unknown)}"},
        )

    await ctx.enqueue({"topic": "config.install", "data": {"_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
