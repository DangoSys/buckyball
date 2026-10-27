from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_target

config = {
    "name": "model-build-api",
    "description": "Build the model recipes of a design",
    "flows": ["model"],
    "triggers": [api("POST", "/model/build")],
    "enqueues": ["model.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        design = require_design(body)
        selection = require_target(body, ("device",))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "model.build",
            "data": {
                "design": design,
                "model": body.get("model"),
                "target": selection,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
