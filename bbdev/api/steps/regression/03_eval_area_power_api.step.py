import os

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip
from utils.path import check_dc_rtl_args, get_buckyball_path

config = {
    "name": "regression-eval-area-power-api",
    "description": "DC area+freq for regression",
    "flows": ["regression"],
    "triggers": [api("POST", "/regression/eval-area-power")],
    "enqueues": ["regression.eval-area-power"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
        subset = {"chip": chip}
        if "top" in body:
            subset["top"] = body.get("top")
        check_dc_rtl_args(subset)
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})

    bbdir = get_buckyball_path()
    tapeout = os.path.join(bbdir, "examples", "chips", chip, "tapeout")
    if not os.path.isdir(tapeout):
        return ApiResponse(
            status=400,
            body={"error": f"chip {chip} has no tapeout directory: {tapeout}"},
        )

    data = {
        "chip": chip,
        "_trace_id": ctx.trace_id,
    }
    top = body.get("top")
    if isinstance(top, str) and top:
        data["top"] = top

    await ctx.enqueue({"topic": "regression.eval-area-power", "data": data})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
