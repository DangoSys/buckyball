from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

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
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
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
        "jobs": jobs,
    }
    await ctx.enqueue(
        {
            "topic": "vcs.build",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
