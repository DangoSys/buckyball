from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir

config = {
    "name": "bebop-verilator-build-api",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/build")],
    "enqueues": ["bebop.verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    unsupported = [name for name in ("vsrc_dir", "output_dir") if name in body]
    if unsupported:
        return ApiResponse(
            status=400,
            body={"error": f"Unsupported parameter(s): {', '.join(unsupported)}. Use --vsrc-dir/--output-dir."},
        )

    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    rushb = bool(body.get("rushB", False))
    data = {
        "chip": chip,
        "jobs": body.get("jobs", 16),
        "diff": bool(body.get("diff", False)),
        "rushB": rushb,
        "vsrc_dir": rtl_dir(
            bbdir,
            chip,
            "verilog",
            body.get("vsrc-dir") or body.get("output-dir"),
            rushb=rushb,
        ),
    }
    await ctx.enqueue({"topic": "bebop.verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
