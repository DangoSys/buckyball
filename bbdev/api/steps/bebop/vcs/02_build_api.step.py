from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "bebop-vcs-build-api",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/vcs/build")],
    "enqueues": ["vcs.build"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "vcs.build",
            "data": {
                "design": require_design(body),
                "jobs": body.get("jobs", 16),
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
