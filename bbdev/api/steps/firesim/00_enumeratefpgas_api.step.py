from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "firesim-enumeratefpgas-api",
    "description": "enumerate FPGAs",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/enumeratefpgas")],
    "enqueues": ["firesim.enumeratefpgas"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "firesim.enumeratefpgas", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
