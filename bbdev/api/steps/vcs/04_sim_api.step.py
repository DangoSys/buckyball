from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_product

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
        selection = require_target(body, ("device", "hart"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
        require_product(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
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
