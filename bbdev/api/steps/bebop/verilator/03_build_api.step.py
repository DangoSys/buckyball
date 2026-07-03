from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-verilator-build-api",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/build")],
    "enqueues": ["bebop.verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    arch_config = body.get("config")
    if not isinstance(arch_config, str) or not arch_config or arch_config == "None":
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"}
        )

    data = {
        "config": arch_config,
    }
    if "vsrc_dir" in body:
        data["vsrc_dir"] = body.get("vsrc_dir")
    await ctx.enqueue({"topic": "bebop.verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
