from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path

config = {
    "name": "verilator-clean-api",
    "description": "clean build directory",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/clean")],
    "enqueues": ["verilator.clean"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    await ctx.enqueue({
        "topic": "verilator.clean",
        "data": {**body, "chip": chip, "task": "clean", "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
