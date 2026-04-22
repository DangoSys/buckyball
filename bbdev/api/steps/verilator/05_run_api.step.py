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

    data = {
        "binary": body.get("binary", ""),
        "config": body.get("config", "sims.verilator.BuckyballToyVerilatorConfig"),
        "jobs": body.get("jobs", "16"),
        "batch": body.get("batch", False),
        "from_run_workflow": True,
    }

    await ctx.enqueue({"topic": "verilator.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
