from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-bemu-sim-api",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/sim")],
    "enqueues": ["bebop.bemu.sim"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    binary = body.get("binary", "")
    if not binary:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is required",
            },
        )

    await ctx.enqueue({"topic": "bebop.bemu.sim", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
