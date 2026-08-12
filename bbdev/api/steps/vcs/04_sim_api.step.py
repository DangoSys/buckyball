from motia import ApiRequest, ApiResponse, FlowContext, api


config = {
    "name": "vcs-sim-api",
    "description": "run a built VCS BBSimHarness simulation",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/sim")],
    "enqueues": ["vcs.sim"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    config_name = body.get("config")
    binary = body.get("binary")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        return ApiResponse(status=400, body={"error": "Missing required parameter: --config must be specified"})
    if not isinstance(binary, str) or not binary:
        return ApiResponse(status=400, body={"error": "Missing required parameter: --binary must be specified"})
    await ctx.enqueue({"topic": "vcs.sim", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
