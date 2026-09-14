from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "firesim-runworkload-api",
    "description": "run workload",
    "flows": ["firesim"],
    "triggers": [api("POST", "/firesim/runworkload")],
    "enqueues": ["firesim.runworkload"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    data = {"chip": chip, "_trace_id": ctx.trace_id}
    if body.get("jobs") is not None:
        data["jobs"] = body["jobs"]
    await ctx.enqueue({"topic": "firesim.runworkload", "data": data})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
