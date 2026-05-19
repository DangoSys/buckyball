from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-verilator-clean-api",
    "description": "Clean verilator build directory",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/clean")],
    "enqueues": ["bebop.verilator.clean"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}

    config_name = body.get("config")
    if not config_name:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"}
        )

    build_dir = get_verilator_build_dir(bbdir, config_name, body.get("output_dir"))

    data = {
        "config": config_name,
        "output_dir": build_dir,
    }
    await ctx.enqueue({"topic": "bebop.verilator.clean", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
