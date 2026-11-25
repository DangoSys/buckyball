from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design, require_product, require_target

config = {
    "name": "bebop-vcs-sim-api",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/vcs/sim")],
    "enqueues": ["vcs.sim"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        target = require_target(body, ("device", "hart"))
        design = require_design(body)
        require_product(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "vcs.sim",
            "data": {
                "target": target,
                "design": design,
                **{
                    key: body[key]
                    for key in ("model", "bench", "os", "test")
                    if key in body
                },
                "wave": body.get("wave", False),
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
