from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "firesim-runworkload-api",
    "description": "run workload",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/runworkload")],
    "enqueues": ["firesim.runworkload"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    data = {"jobs": body.get("jobs", 16)}
    await ctx.enqueue({"topic": "firesim.runworkload", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
