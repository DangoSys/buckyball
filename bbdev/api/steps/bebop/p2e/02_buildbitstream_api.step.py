from motia import ApiRequest, ApiResponse, FlowContext, api
from pathlib import Path

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir

config = {
    "name": "bebop-p2e-buildbitstream-api",
    "description": "Build Bebop P2E runtime case",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/buildbitstream")],
    "enqueues": ["bebop.p2e.buildbitstream"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    vsrc_dir = rtl_dir(
        bbdir, chip, "p2e", body.get("vsrc_dir") or body.get("vsrc-dir")
    )

    stop_after = body.get("stop_after", body.get("stop-after"))
    if stop_after is not None and stop_after not in ("vsyn", "vcom"):
        return ApiResponse(status=400, body={"error": "stop-after must be vsyn or vcom"})
    output_dir = body.get("output_dir") or body.get("output-dir")
    if stop_after and output_dir and Path(output_dir).exists():
        return ApiResponse(status=400, body={"error": "Resource assessment requires a fresh output_dir"})
    data = {
        "chip": chip,
        "diff": bool(body.get("diff", False)),
        "itrace": bool(body.get("itrace", False)),
        "mtrace": bool(body.get("mtrace", False)),
        "vsrc_dir": vsrc_dir,
        "output_dir": output_dir,
        "stop_after": stop_after,
    }
    await ctx.enqueue({
        "topic": "bebop.p2e.buildbitstream",
        "data": {**data, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
