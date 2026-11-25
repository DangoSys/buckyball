from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "config-install-api",
    "description": "install the design.pb of one config",
    "flows": ["config"],
    "triggers": [api("POST", "/config/install")],
    "enqueues": ["config.install"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    unknown = set(body) - {"config", "device", "hart", "thread", "ball"}
    if unknown:
        return ApiResponse(
            status=400, body={"error": f"Unexpected parameters: {sorted(unknown)}"}
        )
    await ctx.enqueue(
        {"topic": "config.install", "data": {**body, "_trace_id": ctx.trace_id}}
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
