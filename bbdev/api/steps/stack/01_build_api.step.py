from utils.event_common import require_target
import os
import re
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

config = {
    "name": "build-stack-api",
    "description": "build one level of a design's stack",
    "flows": ["stack"],
    "triggers": [api("POST", "/stack/build")],
    "enqueues": ["stack.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    if set(body) - {"design", "device", "core"}:
        raise ValueError(
            "stack build takes --design and --device <config> or --core <entry>"
        )
    if not isinstance(body.get("design"), str):
        raise ValueError("--design requires a design name")
    await ctx.enqueue(
        {
            "topic": "stack.build",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
