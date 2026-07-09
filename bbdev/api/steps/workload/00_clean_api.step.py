from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "workload-clean-api",
    "description": "clean workload output directory",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/clean")],
    "enqueues": ["workload.clean"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if body:
        unknown = ", ".join(sorted(body))
        return ApiResponse(status=400, body={"error": f"Unknown workload clean parameter(s): {unknown}"})

    await ctx.enqueue({"topic": "workload.clean", "data": {"_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
