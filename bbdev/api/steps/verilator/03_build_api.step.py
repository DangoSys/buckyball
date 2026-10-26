from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "verilator-build-api",
    "description": "build verilator executable",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/build")],
    "enqueues": ["verilator.build"],
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
    data = {
        "design": design,
        "jobs": body.get("jobs", 16),
    }
    await ctx.enqueue(
        {
            "topic": "verilator.build",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
