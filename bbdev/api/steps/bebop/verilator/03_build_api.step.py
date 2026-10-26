from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design

config = {
    "name": "bebop-verilator-build-api",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/build")],
    "enqueues": ["bebop.verilator.build"],
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
        "diff": bool(body.get("diff", False)),
    }
    await ctx.enqueue(
        {
            "topic": "bebop.verilator.build",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
