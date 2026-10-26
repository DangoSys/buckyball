from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_target

config = {
    "name": "bebop-vcs-verilog-api",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/vcs/verilog")],
    "enqueues": ["vcs.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    await ctx.enqueue(
        {
            "topic": "vcs.verilog",
            "data": {
                "design": require_design(req.body or {}),
                "target": require_target(req.body or {}),
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
