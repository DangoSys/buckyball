from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "uvm-run-api",
    "description": "Build and run a Ball UVM simulation",
    "flows": ["uvm"],
    "triggers": [api("POST", "/uvm/run")],
    "enqueues": ["uvm.run"],
}


def check_args(body: dict) -> str | None:
    allowed = {"ball", "filelist", "test"}
    for key in body:
        if key not in allowed:
            return f"Unexpected parameter: --{key}"
    ball = body.get("ball")
    if not ball or ball is True:
        return "Missing required parameter: --ball=<name>"
    if body.get("filelist") is True:
        return "Parameter --filelist requires a path value"
    if body.get("test") is True:
        return "Parameter --test requires a value"
    return None


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    error = check_args(body)
    if error:
        return ApiResponse(status=400, body={"error": error})

    await ctx.enqueue({"topic": "uvm.run", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
