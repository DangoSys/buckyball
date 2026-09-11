from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {"name": "bebop-vcs-batch-api", "flows": ["bebop"], "triggers": [api("POST", "/bebop/vcs/batch")], "enqueues": ["bebop.vcs.batch"]}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "bebop.vcs.batch", "data": {"chip": require_chip(body), "test": body.get("test", "elf-tests"), "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
