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
    "name": "workload-build-api",
    "description": "build workload",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/build")],
    "enqueues": ["workload.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    allowed = {"chip", "model", "stable", "rushB", "ctest", "mlirtest"}
    unknown = sorted(k for k in body if k not in allowed)
    if unknown:
        return ApiResponse(
            status=400,
            body={"error": f"Unknown workload build parameter(s): {', '.join(unknown)}"},
        )
    chip = body.get("chip")
    if not chip:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --chip must be specified"},
        )
    if not isinstance(chip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        return ApiResponse(
            status=400,
            body={"error": f"Invalid chip: {chip}"},
        )
    chip_dir = Path(get_buckyball_path()) / "examples" / "chips" / chip
    if not chip_dir.is_dir():
        return ApiResponse(
            status=400,
            body={"error": f"Workload chip does not exist: {chip}"},
        )
    stable = body.get("stable", False)
    if not isinstance(stable, bool):
        return ApiResponse(
            status=400,
            body={"error": "Invalid parameter: stable must be a boolean flag"},
        )
    rushb_backend = body.get("rushB")
    if rushb_backend is not None and rushb_backend not in {"bemu", "verilator"}:
        return ApiResponse(
            status=400,
            body={"error": "Invalid --rushB: expected bemu or verilator"},
        )
    ctest = body.get("ctest", False)
    mlirtest = body.get("mlirtest", False)
    if not isinstance(ctest, bool) or not isinstance(mlirtest, bool):
        return ApiResponse(
            status=400,
            body={"error": "--ctest and --mlirtest must be boolean flags"},
        )
    if ctest and mlirtest:
        return ApiResponse(
            status=400,
            body={"error": "--ctest and --mlirtest cannot be used together"},
        )
    if (ctest or mlirtest) and body.get("model"):
        return ApiResponse(
            status=400,
            body={"error": "--ctest and --mlirtest cannot be used with --model"},
        )
    if (ctest or mlirtest) and rushb_backend:
        return ApiResponse(
            status=400,
            body={"error": "--ctest and --mlirtest cannot be used with --rushB"},
        )
    data = {
        "chip": chip,
        "model": body.get("model", ""),
        "stable": stable,
        "rushB": rushb_backend,
        "ctest": ctest,
        "mlirtest": mlirtest,
    }
    await ctx.enqueue({"topic": "workload.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
