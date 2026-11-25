from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_product

config = {
    "name": "bebop-verilator-sim-api",
    "description": "Run bebop verilator simulation",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/sim")],
    "enqueues": ["bebop.verilator.sim"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "hart", "thread"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
        require_product(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {**body, "design": design, "_trace_id": ctx.trace_id}
    await ctx.enqueue(
        {
            "topic": "bebop.verilator.sim",
            "data": {**(data), "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
