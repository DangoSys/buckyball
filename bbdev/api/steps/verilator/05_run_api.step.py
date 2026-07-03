from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "verilator-run-api",
    "description": "trigger complete verilator workflow",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/run")],
    "enqueues": ["verilator.run"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    config_name = body.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"},
        )
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

    data = {
        "binary": binary,
        "config": config_name,
        "jobs": body.get("jobs", "16"),
        "batch": body.get("batch", False),
        "from_run_workflow": True,
    }

    await ctx.enqueue({"topic": "verilator.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
