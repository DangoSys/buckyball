from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "verilator-build-api",
    "description": "build verilator executable",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/build")],
    "enqueues": ["verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}
    data = {
        "jobs": body.get("jobs", 16),
        "config": body.get("config"),
        "output_dir": get_verilator_build_dir(bbdir, body.get("config"), body.get("output_dir")),
    }
    await ctx.enqueue({"topic": "verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
