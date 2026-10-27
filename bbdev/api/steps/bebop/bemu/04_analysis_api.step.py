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

from utils.path import get_buckyball_path, run_log_dir
from bemu_analysis import design_maps

config = {
    "name": "bebop-bemu-analysis-api",
    "description": "Analyze bebop bemu itrace/mtrace",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/analysis")],
    "enqueues": ["bebop.bemu.analysis"],
}


def _fail(message: str, status: int = 400) -> ApiResponse:
    return ApiResponse(
        status=status,
        body={
            "success": False,
            "failure": True,
            "returncode": status,
            "message": message,
        },
    )


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    allowed = {"device", "core", "design", "run", "itrace", "mtrace"}
    unknown = sorted(k for k in body if k not in allowed)
    if unknown:
        return _fail(f"unknown parameter(s): {', '.join(unknown)}")

    design = body.get("design", "")
    if not design:
        return _fail("design parameter is required")
    try:
        design_maps(get_buckyball_path(), design, selection)
    except ValueError as e:
        return _fail(str(e))

    itrace = bool(body.get("itrace", False))
    mtrace = bool(body.get("mtrace", False))
    if not itrace and not mtrace:
        return _fail("need --itrace and/or --mtrace")

    try:
        run_log_dir(get_buckyball_path(), body.get("run"))
    except ValueError as e:
        return _fail(str(e))

    await ctx.enqueue(
        {
            "topic": "bebop.bemu.analysis",
            "data": {
                "design": design,
                "run": body["run"],
                "itrace": itrace,
                "mtrace": mtrace,
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
