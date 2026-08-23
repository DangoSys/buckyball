from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
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
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    await ctx.enqueue({
        "topic": "bebop.p2e.clean",
        "data": {**body, "chip": chip, "task": "clean", "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
