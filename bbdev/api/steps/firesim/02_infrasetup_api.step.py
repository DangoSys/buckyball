from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "firesim-infrasetup-api",
    "description": "infrasetup",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/infrasetup")],
    "enqueues": ["firesim.infrasetup"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
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
            "topic": "firesim.infrasetup",
            "data": {"design": design, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
