from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_vcs_build_dir


config = {
    "name": "vcs-build-api",
    "description": "build a VCS BBSimHarness executable",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/build")],
    "enqueues": ["vcs.build"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    config_name = body.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        return ApiResponse(status=400, body={"error": "Missing required parameter: --config must be specified"})
    jobs = body.get("jobs", 16)
    try:
        jobs = int(jobs)
    except (TypeError, ValueError):
        return ApiResponse(status=400, body={"error": "jobs must be a positive integer"})
    if jobs <= 0:
        return ApiResponse(status=400, body={"error": "jobs must be a positive integer"})
    bbdir = get_buckyball_path()
    data = {
        "config": config_name,
        "jobs": jobs,
        "output_dir": get_vcs_build_dir(bbdir, config_name, body.get("output_dir")),
    }
    await ctx.enqueue({"topic": "vcs.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
