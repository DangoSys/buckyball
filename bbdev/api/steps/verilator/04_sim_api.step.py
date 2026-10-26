from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "verilator-sim-api",
    "description": "run verilator simulation",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/sim")],
    "enqueues": ["verilator.sim"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    binary = body.get("binary", "")
    if not binary:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is required",
            },
        )
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    await ctx.enqueue(
        {
            "topic": "verilator.sim",
            "data": {
                **body,
                "design": design,
                "task": "sim",
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
