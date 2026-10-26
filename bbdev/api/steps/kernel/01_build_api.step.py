from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "kernel-build-api",
    "description": "build RISC-V kernel + rootfs for Pegasus",
    "flows": ["kernel"],
    "triggers": [api("POST", "/kernel/build")],
    "enqueues": ["kernel.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device",))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        require_design(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "kernel.build",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
