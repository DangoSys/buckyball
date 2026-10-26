from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design
from utils.path import get_buckyball_path

config = {
    "name": "bebop-p2e-clean-api",
    "description": "Clean P2E build directory",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/clean")],
    "enqueues": ["bebop.p2e.clean"],
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
            "topic": "bebop.p2e.clean",
            "data": {
                **body,
                "design": design,
                "task": "clean",
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
