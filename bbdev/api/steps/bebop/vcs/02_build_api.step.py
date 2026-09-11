from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {"name": "bebop-vcs-build-api", "flows": ["bebop"], "triggers": [api("POST", "/bebop/vcs/build")], "enqueues": ["vcs.build"]}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "vcs.build", "data": {"chip": require_chip(body), "jobs": body.get("jobs", 16), "bebop_vcs": True, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
