from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import wait_for_result

config = {
    "name": "Pegasus Buildbitstream",
    "description": "build pegasus bitstream",
    "flows": ["pegasus"],
    "triggers": [api("POST", "/pegasus/buildbitstream")],
    "enqueues": ["pegasus.buildbitstream"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue({"topic": "pegasus.buildbitstream", "data": {**body, "_trace_id": ctx.trace_id}})
    result = await wait_for_result(ctx)
    return result
