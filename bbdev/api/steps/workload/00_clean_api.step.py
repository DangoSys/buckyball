from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "workload-clean-api",
    "description": "clean workload output directory for one design",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/clean")],
    "enqueues": ["workload.clean"],
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

    unknown = sorted(k for k in body if k not in {"design", "device", "core"})
    if unknown:
        return ApiResponse(
            status=400,
            body={
                "error": f"Unknown workload clean parameter(s): {', '.join(unknown)}"
            },
        )

    await ctx.enqueue(
        {
            "topic": "workload.clean",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
