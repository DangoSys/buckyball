from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bbmap-designs-api",
    "description": "map every design: what it is for, its models, configs and composition",
    "flows": ["bbmap"],
    "triggers": [api("POST", "/bbmap/designs")],
    "enqueues": ["bbmap.designs"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if body:
        return ApiResponse(
            status=400,
            body={"error": f"bbmap designs takes no parameters, got {sorted(body)}"},
        )
    await ctx.enqueue({"topic": "bbmap.designs", "data": {"_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
