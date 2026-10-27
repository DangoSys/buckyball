from utils.event_common import require_target

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "bebop-vcs-batch-api",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/vcs/batch")],
    "enqueues": ["bebop.vcs.batch"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        selection = require_target(body, ("device",))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "bebop.vcs.batch",
            "data": {
                "design": require_design(body),
                "test": body.get("test", "bare-tests"),
                "jobs": body.get("jobs", 1),
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
