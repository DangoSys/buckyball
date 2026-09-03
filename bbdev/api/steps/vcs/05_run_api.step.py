from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir


config = {
    "name": "vcs-run-api",
    "description": "generate RTL, build, and run VCS",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/run")],
    "enqueues": ["vcs.verilog"],
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
        "binary": binary,
        "batch": bool(body.get("batch", False)),
        "jobs": jobs,
        "output_dir": rtl_dir(bbdir, chip, "tapeout", body.get("output_dir")),
        "from_run_workflow": True,
    }
    await ctx.enqueue({"topic": "vcs.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
