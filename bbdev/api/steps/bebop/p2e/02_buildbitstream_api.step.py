from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design
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
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    vsrc_dir = rtl_dir(
        bbdir,
        design,
        "p2e",
        body.get("vsrc_dir") or body.get("vsrc-dir"),
        selection=selection,
    )

    output_dir = body.get("output_dir") or body.get("output-dir")
    data = {
        "design": design,
        "diff": bool(body.get("diff", False)),
        "itrace": bool(body.get("itrace", False)),
        "mtrace": bool(body.get("mtrace", False)),
        "vsrc_dir": vsrc_dir,
        "output_dir": output_dir,
    }
    await ctx.enqueue(
        {
            "topic": "bebop.p2e.buildbitstream",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
