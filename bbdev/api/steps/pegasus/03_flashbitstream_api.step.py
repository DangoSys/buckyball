from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "Pegasus Flashbitstream",
    "description": "Flash bitstream onto AU280 via hw_server",
    "flows": ["pegasus"],
    "triggers": [api("POST", "/pegasus/flashbitstream")],
    "enqueues": ["pegasus.flashbitstream"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "pegasus.flashbitstream", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
