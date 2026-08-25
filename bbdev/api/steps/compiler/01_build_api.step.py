import os
import re
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
    "name": "build-compiler-api",
    "description": "build compiler",
    "flows": ["compiler"],
    "triggers": [api("POST", "/compiler/build")],
    "enqueues": ["compiler.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if "core" in body:
        raise ValueError("compiler build takes --chip, not --core")
    chip = body.get("chip")
    if not isinstance(chip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        raise ValueError(f"Invalid compiler chip: {chip}")
    await ctx.enqueue({"topic": "compiler.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
