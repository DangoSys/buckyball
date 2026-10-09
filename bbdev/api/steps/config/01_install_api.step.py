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
    unknown = set(body) - {"chip"}
    if unknown:
        return ApiResponse(status=400, body={"error": f"Unexpected parameters: {sorted(unknown)}"})
    if "chip" in body and (not isinstance(body["chip"], str) or not body["chip"].replace("_", "").replace("-", "").isalnum()):
        return ApiResponse(status=400, body={"error": "--chip requires a chip name"})
    await ctx.enqueue({"topic": "config.install", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
