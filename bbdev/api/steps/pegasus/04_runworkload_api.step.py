from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "Pegasus Runworkload",
    "description": "Run workload on AU280 (load Linux image + start CPU + collect UART)",
    "flows": ["pegasus"],
    "triggers": [api("POST", "/pegasus/runworkload")],
    "enqueues": ["pegasus.runworkload"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "pegasus.runworkload", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
