from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "config-install-api",
    "description": "install all design configs under examples/designs",
    "flows": ["config"],
    "triggers": [api("POST", "/config/install")],
    "enqueues": ["config.install"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if "design" in body:
        try:
            require_target(body)
        except ValueError as error:
            return ApiResponse(status=400, body={"error": str(error)})
    unknown = set(body) - {"design", "device", "core"}
    if unknown:
        return ApiResponse(
            status=400, body={"error": f"Unexpected parameters: {sorted(unknown)}"}
        )
    if "design" in body and (
        not isinstance(body["design"], str)
        or not body["design"].replace("_", "").replace("-", "").isalnum()
    ):
        return ApiResponse(
            status=400, body={"error": "--design requires a design name"}
        )
    await ctx.enqueue(
        {"topic": "config.install", "data": {**body, "_trace_id": ctx.trace_id}}
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
