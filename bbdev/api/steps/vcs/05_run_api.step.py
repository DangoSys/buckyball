from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "vcs-run-api",
    "description": "generate RTL, build, and run VCS",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/run")],
    "enqueues": ["vcs.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    binary = body.get("binary")
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    if not isinstance(binary, str) or not binary:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --binary must be specified"},
        )
    jobs = body.get("jobs", 16)
    try:
        jobs = int(jobs)
    except (TypeError, ValueError):
        return ApiResponse(
            status=400, body={"error": "jobs must be a positive integer"}
        )
    if jobs <= 0:
        return ApiResponse(
            status=400, body={"error": "jobs must be a positive integer"}
        )
    data = {
        "design": design,
        "binary": binary,
        "batch": bool(body.get("batch", False)),
        "jobs": jobs,
        "from_run_workflow": True,
    }
    await ctx.enqueue(
        {
            "topic": "vcs.verilog",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
