from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "image-build-api",
    "description": "build RISC-V kernel + rootfs for Pegasus",
    "flows": ["image"],
    "triggers": [api("POST", "/image/build")],
    "enqueues": ["image.build"],
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
    if len([key for key in ("model", "bench", "interactive") if key in body]) > 1:
        return ApiResponse(
            status=400,
            body={"error": "An image runs one of --model, --bench, --interactive"},
        )
    await ctx.enqueue(
        {
            "topic": "image.build",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
