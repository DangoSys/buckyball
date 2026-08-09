import os
import re
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.chip import available_compiler_chips, available_cores, resolve_chip_compiler_core, resolve_core
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

    stable = body.get("stable", False)
    if not isinstance(stable, bool):
        return ApiResponse(
            status=400,
            body={"error": "Invalid parameter: stable must be a boolean flag"},
        )

    chip = body.get("chip")
    core = body.get("core")
    if bool(chip) == bool(core):
        return ApiResponse(
            status=400,
            body={"error": "Specify exactly one compiler target: --core or --chip"},
        )
    target_name = core or chip
    if not isinstance(target_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", target_name):
        return ApiResponse(
            status=400,
            body={"error": f"Invalid compiler target: {target_name}"},
        )

    try:
        if core:
            resolve_core(get_buckyball_path(), core, require_compiler=True)
        else:
            resolve_chip_compiler_core(get_buckyball_path(), chip)
    except ValueError as error:
        choices = available_cores(get_buckyball_path()) if core else available_compiler_chips(get_buckyball_path())
        return ApiResponse(
            status=400,
            body={
                "error": f"{error}; available {'cores' if core else 'chips'} are: {', '.join(choices)}",
            },
        )

    await ctx.enqueue({"topic": "compiler.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
