from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "firesim-infrasetup-api",
    "description": "infrasetup",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/infrasetup")],
    "enqueues": ["firesim.infrasetup"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    data = {"jobs": body.get("jobs", 16)}
    await ctx.enqueue({"topic": "firesim.infrasetup", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
