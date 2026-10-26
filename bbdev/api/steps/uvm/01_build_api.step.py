from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "uvm-build-api",
    "description": "Build a Ball UVM simulation",
    "flows": ["uvm"],
    "triggers": [api("POST", "/uvm/build")],
    "enqueues": ["uvm.build"],
}


def check_args(body: dict) -> str | None:
    allowed = {"design", "device", "ball", "ip"}
    for key in body:
        if key not in allowed:
            return f"Unexpected parameter: --{key}"
    if body.get("design") is True or ("design" in body and not body["design"]):
        return "Parameter --design requires a value"
    if not body.get("design") and not isinstance(body.get("ip"), str):
        return "Standalone UVM requires --ip"
    if body.get("ball") is True:
        return "Parameter --ball requires a value"
    if body.get("ip") is True:
        return "Parameter --ip requires a value"
    if body.get("ball") and body.get("ip"):
        return "Parameters --ball and --ip are mutually exclusive"
    return None


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "ip", "ball"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    error = check_args(body)
    if error:
        return ApiResponse(status=400, body={"error": error})

    await ctx.enqueue(
        {
            "topic": "uvm.build",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
