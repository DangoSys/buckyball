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
    data = {
        "workload": body.get("workload", ""),
        "model": body.get("model", ""),
    }
    await ctx.enqueue({"topic": "workload.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
