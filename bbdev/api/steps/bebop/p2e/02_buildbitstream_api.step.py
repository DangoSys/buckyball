from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-p2e-buildbitstream-api",
    "description": "Build Bebop P2E runtime case",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/buildbitstream")],
    "enqueues": ["bebop.p2e.buildbitstream"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    config_name = body.get("config", "sims.p2e.P2EToyConfig")
    vsrc_dir = get_verilator_build_dir(bbdir, config_name, body.get("vsrc_dir"))

    data = {
        "config": config_name,
        "vsrc_dir": vsrc_dir,
        "output_dir": body.get("output_dir"),
    }
    await ctx.enqueue({
        "topic": "bebop.p2e.buildbitstream",
        "data": {**data, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
