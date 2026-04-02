from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "verilator-sim-api",
    "description": "run verilator simulation",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/sim")],
    "enqueues": ["verilator.sim"],
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

    await ctx.enqueue({"topic": "verilator.sim", "data": {**body, "task": "sim", "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
