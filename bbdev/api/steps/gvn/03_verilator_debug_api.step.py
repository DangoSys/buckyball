from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_target

config = {
    "name": "gvn-verilator-debug-api",
    "description": "Check a model's GVN slices on Verilator against their BEMU references",
    "flows": ["gvn"],
    "triggers": [api("POST", "/gvn/verilator/debug")],
    "enqueues": ["gvn.verilator.debug"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device",))
        design = require_design(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "gvn.verilator.debug",
            "data": {
                **body,
                "design": design,
                "target": selection,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
