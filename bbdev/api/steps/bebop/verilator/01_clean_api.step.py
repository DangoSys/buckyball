from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir

config = {
    "name": "bebop-verilator-clean-api",
    "description": "Clean verilator build directory",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/clean")],
    "enqueues": ["bebop.verilator.clean"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    rushb = bool(body.get("rushB", False))
    build_dir = rtl_dir(
        bbdir, chip, "verilog", body.get("output_dir"), rushb=rushb
    )
    data = {
        "chip": chip,
        "output_dir": build_dir,
        "rushB": rushb,
    }
    await ctx.enqueue({"topic": "bebop.verilator.clean", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
