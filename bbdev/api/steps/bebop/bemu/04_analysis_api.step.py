import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import get_buckyball_path
from bemu_analysis import abs_log_dir, chip_maps

config = {
    "name": "bebop-bemu-analysis-api",
    "description": "Analyze bebop bemu itrace/mtrace",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/analysis")],
    "enqueues": ["bebop.bemu.analysis"],
}

ALLOWED = {"chip", "log-dir", "log_dir", "itrace", "mtrace"}


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
    unknown = sorted(k for k in body if k not in ALLOWED)
    if unknown:
        return _fail(f"unknown parameter(s): {', '.join(unknown)}")

    chip = body.get("chip", "")
    if not chip:
        return _fail("chip parameter is required")
    try:
        chip_maps(get_buckyball_path(), chip)
    except ValueError as e:
        return _fail(str(e))

    itrace = bool(body.get("itrace", False))
    mtrace = bool(body.get("mtrace", False))
    if not itrace and not mtrace:
        return _fail("need --itrace and/or --mtrace")

    log_dir = body.get("log-dir", body.get("log_dir", ""))
    try:
        log_dir = str(abs_log_dir(log_dir))
    except ValueError as e:
        return _fail(str(e))

    await ctx.enqueue(
        {
            "topic": "bebop.bemu.analysis",
            "data": {
                "chip": chip,
                "log-dir": log_dir,
                "itrace": itrace,
                "mtrace": mtrace,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
