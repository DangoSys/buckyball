import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.bemu import validate_bemu_chip
from utils.path import get_buckyball_path

config = {
    "name": "bebop-bemu-sim-api",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/sim")],
    "enqueues": ["bebop.bemu.sim"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    chip = body.get("chip", "")
    try:
        validate_bemu_chip(chip, get_buckyball_path())
    except ValueError as e:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": str(e),
            },
        )

    binary = body.get("binary", "")
    if not binary:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is required",
            },
        )

    await ctx.enqueue({"topic": "bebop.bemu.sim", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
