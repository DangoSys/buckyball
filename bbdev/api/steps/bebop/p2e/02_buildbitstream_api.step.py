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

    resume = body.get("resume_post_route", body.get("resume-post-route", False))
    if not isinstance(resume, bool):
        return ApiResponse(status=400, body={"error": "resume-post-route must be a boolean flag"})
    stop_after = body.get("stop_after", body.get("stop-after"))
    if stop_after is not None and stop_after not in ("vsyn", "vcom"):
        return ApiResponse(status=400, body={"error": "stop-after must be vsyn or vcom"})
    if resume and stop_after is not None:
        return ApiResponse(status=400, body={"error": "stop-after conflicts with resume-post-route"})
    output_dir = body.get("output_dir") or body.get("output-dir")
    if stop_after and output_dir and Path(output_dir).exists():
        return ApiResponse(status=400, body={"error": "Resource assessment requires a fresh output_dir"})
    if resume:
        if not isinstance(output_dir, str) or not output_dir:
            return ApiResponse(status=400, body={"error": "Post-route resume requires output_dir"})
        case = Path(output_dir).resolve()
        build_root = (Path(bbdir) / "bebop/build").resolve()
        if build_root not in case.parents or not case.is_dir():
            return ApiResponse(status=400, body={"error": "P2E resume requires an existing case under bebop/build"})
        for artifact in ("xepic_vvac_top_0_0_route.dcp", "xepic_vvac_top_0_0.bit"):
            path = case / "fpgaCompDir/part_b0_f0/pnrDir" / artifact
            if not path.is_file():
                return ApiResponse(status=400, body={"error": f"Post-route resume requires {path}"})

    data = {
        "chip": chip,
        "diff": bool(body.get("diff", False)),
        "vsrc_dir": vsrc_dir,
        "output_dir": output_dir,
        "resume_post_route": resume,
        "stop_after": stop_after,
    }
    await ctx.enqueue({
        "topic": "bebop.p2e.buildbitstream",
        "data": {**data, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
