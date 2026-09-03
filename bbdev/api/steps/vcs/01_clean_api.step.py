from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir


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
                "output_dir": rtl_dir(bbdir, chip, "tapeout", body.get("output_dir")),
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
