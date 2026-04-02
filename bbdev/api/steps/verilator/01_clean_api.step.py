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
    await ctx.enqueue({"topic": "verilator.clean", "data": {**body, "task": "clean", "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
