from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
from utils.path import get_buckyball_path, get_p2e_build_dir

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
    vsrc_dir = get_p2e_build_dir(bbdir, chip, body.get("vsrc_dir"))

    data = {
        "chip": chip,
        "vsrc_dir": vsrc_dir,
        "output_dir": body.get("output_dir"),
    }
    await ctx.enqueue({
        "topic": "bebop.p2e.buildbitstream",
        "data": {**data, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
