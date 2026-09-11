from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {"name": "bebop-vcs-verilog-api", "flows": ["bebop"], "triggers": [api("POST", "/bebop/vcs/verilog")], "enqueues": ["vcs.verilog"]}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    await ctx.enqueue({"topic": "vcs.verilog", "data": {"chip": require_chip(req.body or {}), "bebop_vcs": True, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
