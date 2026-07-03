from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "verilator-clean-api",
    "description": "clean build directory",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/clean")],
    "enqueues": ["verilator.clean"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    config_name = body.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"},
        )
    await ctx.enqueue({"topic": "verilator.clean", "data": {**body, "task": "clean", "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
