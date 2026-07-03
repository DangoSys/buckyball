from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "workload-build-api",
    "description": "build workload",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/build")],
    "enqueues": ["workload.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    allowed = {"model", "stable"}
    unknown = sorted(k for k in body if k not in allowed)
    if unknown:
        return ApiResponse(
            status=400,
            body={"error": f"Unknown workload build parameter(s): {', '.join(unknown)}"},
        )
    stable = body.get("stable", False)
    if not isinstance(stable, bool):
        return ApiResponse(
            status=400,
            body={"error": "Invalid parameter: stable must be a boolean flag"},
        )
    data = {
        "model": body.get("model", ""),
        "stable": stable,
    }
    await ctx.enqueue({"topic": "workload.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
