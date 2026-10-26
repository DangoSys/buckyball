from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design, require_target

config = {
    "name": "bebop-vcs-sim-api",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/vcs/sim")],
    "enqueues": ["vcs.sim"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    await ctx.enqueue(
        {
            "topic": "vcs.sim",
            "data": {
                "design": require_design(body),
                "target": require_target(body, ("device", "core")),
                "binary": body["binary"],
                "wave": body.get("wave", False),
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
