from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_vcs_build_dir


config = {
    "name": "vcs-verilog-api",
    "description": "generate BBSimHarness RTL for VCS",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/verilog")],
    "enqueues": ["vcs.verilog"],
}


def _request_data(body: dict) -> dict:
    config_name = body.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        raise ValueError("Missing required parameter: --config must be specified")
    bbdir = get_buckyball_path()
    return {
        "config": config_name,
        "output_dir": get_vcs_build_dir(bbdir, config_name, body.get("output_dir")),
    }


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    try:
        data = _request_data(req.body or {})
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})
    await ctx.enqueue({"topic": "vcs.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
