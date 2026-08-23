from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
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
    bbdir = get_buckyball_path()
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    await ctx.enqueue(
        {
            "topic": "vcs.clean",
            "data": {
                "chip": chip,
                "output_dir": get_vcs_build_dir(bbdir, chip, body.get("output_dir")),
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
