from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir


config = {
    "name": "vcs-build-api",
    "description": "build a VCS BBSimHarness executable",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/build")],
    "enqueues": ["vcs.build"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    jobs = body.get("jobs", 16)
    try:
        jobs = int(jobs)
    except (TypeError, ValueError):
        return ApiResponse(status=400, body={"error": "jobs must be a positive integer"})
    if jobs <= 0:
        return ApiResponse(status=400, body={"error": "jobs must be a positive integer"})
    bbdir = get_buckyball_path()
    data = {
        "chip": chip,
        "jobs": jobs,
        "output_dir": rtl_dir(bbdir, chip, "tapeout", body.get("output_dir")),
    }
    await ctx.enqueue({"topic": "vcs.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
