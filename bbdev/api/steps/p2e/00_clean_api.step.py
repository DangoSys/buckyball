from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "p2e-clean-api",
    "description": "Clean P2E build directory",
    "flows": ["p2e"],
    "triggers": [api("POST", "/p2e/clean")],
    "enqueues": ["p2e.clean"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "p2e.clean", "data": {**body, "task": "clean", "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})

