from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "verilator-build-api",
    "description": "build verilator executable",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/build")],
    "enqueues": ["verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    data = {
        "jobs": body.get("jobs", 16),
        "cosim": body.get("cosim", False),
    }
    await ctx.enqueue({"topic": "verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
