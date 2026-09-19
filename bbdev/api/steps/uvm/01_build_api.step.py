from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "uvm-build-api",
    "description": "Build a Ball UVM simulation",
    "flows": ["uvm"],
    "triggers": [api("POST", "/uvm/build")],
    "enqueues": ["uvm.build"],
}


def check_args(body: dict) -> str | None:
    allowed = {"chip", "ball", "ip"}
    for key in body:
        if key not in allowed:
            return f"Unexpected parameter: --{key}"
    if not body.get("chip") or body.get("chip") is True:
        return "Missing required parameter: --chip"
    if body.get("ball") is True:
        return "Parameter --ball requires a value"
    if body.get("ip") is True:
        return "Parameter --ip requires a value"
    if body.get("ball") and body.get("ip"):
        return "Parameters --ball and --ip are mutually exclusive"
    return None


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    error = check_args(body)
    if error:
        return ApiResponse(status=400, body={"error": error})

    await ctx.enqueue({"topic": "uvm.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
