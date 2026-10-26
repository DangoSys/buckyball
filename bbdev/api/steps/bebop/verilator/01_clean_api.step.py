from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design

config = {
    "name": "bebop-verilator-clean-api",
    "description": "Clean verilator build directory",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/clean")],
    "enqueues": ["bebop.verilator.clean"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    data = {"design": design}
    await ctx.enqueue(
        {
            "topic": "bebop.verilator.clean",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
