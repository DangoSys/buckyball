import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import require_chip
from utils.path import get_buckyball_path
from steps.bebop.bemu.scripts.model_sim import model_run_commands

config = {
    "name": "bebop-bemu-sim-api",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/sim")],
    "enqueues": ["bebop.bemu.sim"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if "pk" in body or "host-io" in body:
        return ApiResponse(
            status=400, body={"error": "Unknown parameter: pk or host-io"}
        )
    if "system" in body:
        return ApiResponse(status=400, body={"error": "Unknown parameter: system"})
    if body.get("core_index") is not None and any(
        key in body for key in ("dtb", "initrd", "memory-mib", "load-manifest")
    ):
        return ApiResponse(
            status=400, body={"error": "chip boot options cannot select a single core"}
        )
    if body.get("load-manifest") and any(
        key in body for key in ("dtb", "initrd", "memory-mib")
    ):
        return ApiResponse(
            status=400,
            body={"error": "load-manifest conflicts with dtb, initrd and memory-mib"},
        )
    chip = body.get("chip", "")
    try:
        require_chip({"chip": chip})
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
    if body.get("model"):
        try:
            model_run_commands(get_buckyball_path(), body)
        except (ValueError, KeyError, OSError) as error:
            return ApiResponse(
                status=400,
                body={
                    "success": False,
                    "failure": True,
                    "returncode": 400,
                    "message": str(error),
                },
            )
    elif not binary:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "model or binary parameter is required",
            },
        )

    await ctx.enqueue(
        {"topic": "bebop.bemu.sim", "data": {**body, "_trace_id": ctx.trace_id}}
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
