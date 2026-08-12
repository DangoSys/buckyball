from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_vcs_build_dir


config = {
    "name": "vcs-clean-api",
    "description": "clean VCS artifacts without deleting generated RTL",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/clean")],
    "enqueues": ["vcs.clean"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    config_name = body.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        return ApiResponse(status=400, body={"error": "Missing required parameter: --config must be specified"})
    bbdir = get_buckyball_path()
    await ctx.enqueue(
        {
            "topic": "vcs.clean",
            "data": {
                "config": config_name,
                "output_dir": get_vcs_build_dir(bbdir, config_name, body.get("output_dir")),
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
