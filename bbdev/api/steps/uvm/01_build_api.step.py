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
    allowed = {"config", "device", "ball", "ip"}
    for key in body:
        if key not in allowed:
            return f"Unexpected parameter: --{key}"
    for key in allowed:
        if body.get(key) is True:
            return f"Parameter --{key} requires a value"
    if bool(body.get("ball")) == bool(body.get("ip")):
        return "Specify exactly one of --ball <name> and --ip <name>"
    if body.get("ball") and ("config" in body or "device" in body):
        return (
            "Ball UVM runs the Ball alone: --ball <name> takes no --config or --device"
        )
    return None


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    error = check_args(body)
    if error:
        return ApiResponse(status=400, body={"error": error})
    # A Ball is verified alone, from its default config; an IP through one
    # of a design's device configs.
    selection = None
    try:
        if body.get("ball"):
            target = {"ball": body["ball"]}
            selection = require_target(target, ("ball",))
            body["design"] = target["design"]
        elif "device" in body:
            target = {k: body[k] for k in ("config", "device") if k in body}
            selection = require_target(target, ("device",))
            body["design"] = target["design"]
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})

    await ctx.enqueue(
        {
            "topic": "uvm.build",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
