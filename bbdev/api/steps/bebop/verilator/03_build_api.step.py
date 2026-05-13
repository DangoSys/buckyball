from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-verilator-build-api",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/build")],
    "enqueues": ["bebop.verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    arch_config = body.get("config", "sims.verilator.BuckyballToyVerilatorConfig")
    vsrc_dir = get_verilator_build_dir(bbdir, arch_config, body.get("vsrc_dir"))

    data = {
        "config": arch_config,
        "vsrc_dir": vsrc_dir,
    }
    await ctx.enqueue({"topic": "bebop.verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
