from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
from utils.path import get_buckyball_path


config = {
    "name": "vcs-sim-api",
    "description": "run a built VCS BBSimHarness simulation",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/sim")],
    "enqueues": ["vcs.sim"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    binary = body.get("binary")
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    if not isinstance(binary, str) or not binary:
        return ApiResponse(status=400, body={"error": "Missing required parameter: --binary must be specified"})
    await ctx.enqueue({
        "topic": "vcs.sim",
        "data": {**body, "chip": chip, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
