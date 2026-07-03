import os
import re
import sys
from pathlib import Path

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


def available_chips() -> list[str]:
    chips_dir = Path(get_buckyball_path()) / "examples" / "chips"
    return sorted(
        path.name
        for path in chips_dir.iterdir()
        if (path / "compiler" / "CMakeLists.txt").is_file()
    )


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    stable = body.get("stable", False)
    if not isinstance(stable, bool):
        return ApiResponse(
            status=400,
            body={"error": "Invalid parameter: stable must be a boolean flag"},
        )

    chip = body.get("chip")
    if not chip:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --chip must be specified"},
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        return ApiResponse(
            status=400,
            body={"error": f"Invalid chip: {chip}"},
        )

    chips = available_chips()
    chip_dir = Path(get_buckyball_path()) / "examples" / "chips" / chip / "compiler"
    if not (chip_dir / "CMakeLists.txt").is_file():
        return ApiResponse(
            status=400,
            body={
                "error": f"Compiler chip does not exist: {chip}; available compiler chips are: {', '.join(chips)}",
            },
        )

    await ctx.enqueue({"topic": "compiler.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
