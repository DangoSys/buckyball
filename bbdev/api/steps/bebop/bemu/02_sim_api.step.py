from utils.event_common import require_target
import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import require_design

config = {
    "name": "bebop-bemu-sim-api",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/sim")],
    "enqueues": ["bebop.bemu.sim"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
        require_design(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    if not body.get("binary"):
        return ApiResponse(status=400, body={"error": "binary parameter is required"})

    await ctx.enqueue(
        {
            "topic": "bebop.bemu.sim",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
