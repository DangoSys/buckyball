from utils.event_common import require_target
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
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    allowed = {"device", "core", "design", "ctest", "mlirtest", "soctest"}
    unknown = sorted((k for k in body if k not in allowed))
    if unknown:
        return ApiResponse(
            status=400,
            body={
                "error": f"Unknown workload build parameter(s): {', '.join(unknown)}"
            },
        )
    design = body.get("design")
    if not design:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --design must be specified"},
        )
    if not isinstance(design, str) or not re.fullmatch("[A-Za-z0-9_-]+", design):
        return ApiResponse(status=400, body={"error": f"Invalid design: {design}"})
    design_dir = Path(get_buckyball_path()) / "examples" / "designs" / design
    if not design_dir.is_dir():
        return ApiResponse(
            status=400, body={"error": f"Workload design does not exist: {design}"}
        )
    ctest = body.get("ctest", False)
    mlirtest = body.get("mlirtest", False)
    soctest = body.get("soctest", False)
    if not all(isinstance(flag, bool) for flag in (ctest, mlirtest, soctest)):
        return ApiResponse(
            status=400,
            body={"error": "--ctest, --mlirtest and --soctest must be boolean flags"},
        )
    if sum((ctest, mlirtest, soctest)) > 1:
        return ApiResponse(
            status=400,
            body={"error": "--ctest, --mlirtest and --soctest are mutually exclusive"},
        )
    data = {
        "design": design,
        "ctest": ctest,
        "mlirtest": mlirtest,
        "soctest": soctest,
    }
    await ctx.enqueue(
        {
            "topic": "workload.build",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
