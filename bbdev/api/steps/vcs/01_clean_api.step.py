from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "vcs-clean-api",
    "description": "clean VCS artifacts without deleting generated RTL",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/clean")],
    "enqueues": ["vcs.clean"],
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
    await ctx.enqueue(
        {
            "topic": "vcs.clean",
            "data": {"design": design, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
