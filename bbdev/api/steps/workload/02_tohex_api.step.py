from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "workload-tohex-api",
    "description": "convert design workload ELF to hex",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/tohex")],
    "enqueues": ["workload.tohex"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})

    await ctx.enqueue(
        {
            "topic": "workload.tohex",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
