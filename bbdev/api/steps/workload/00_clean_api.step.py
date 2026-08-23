from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip

config = {
    "name": "workload-clean-api",
    "description": "clean workload output directory for one chip",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/clean")],
    "enqueues": ["workload.clean"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})

    unknown = sorted(k for k in body if k not in {"chip"})
    if unknown:
        return ApiResponse(
            status=400,
            body={"error": f"Unknown workload clean parameter(s): {', '.join(unknown)}"},
        )

    await ctx.enqueue({"topic": "workload.clean", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
