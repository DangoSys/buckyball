from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

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
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    data = {"design": design, "_trace_id": ctx.trace_id}
    if body.get("jobs") is not None:
        data["jobs"] = body["jobs"]
    await ctx.enqueue(
        {"topic": "firesim.runworkload", "data": {**(data), "target": selection}}
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
