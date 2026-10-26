from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "vcs-sim-api",
    "description": "run a built VCS BBSimHarness simulation",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/sim")],
    "enqueues": ["vcs.sim"],
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
    await ctx.enqueue(
        {
            "topic": "vcs.sim",
            "data": {
                **body,
                "design": design,
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
