from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-verilator-sim-api",
    "description": "Run bebop verilator simulation",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/sim")],
    "enqueues": ["bebop.verilator.sim"],
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

    await ctx.enqueue({"topic": "bebop.verilator.sim", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
