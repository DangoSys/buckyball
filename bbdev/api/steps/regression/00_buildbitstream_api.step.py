from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "regression-buildbitstream-api",
    "description": "Chain p2e verilog and buildbitstream for regression",
    "flows": ["regression"],
    "triggers": [api("POST", "/regression/buildbitstream")],
    "enqueues": ["regression.buildbitstream"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})

    data = {
        "chip": chip,
        "_trace_id": ctx.trace_id,
    }
    output_dir = body.get("output_dir")
    if output_dir is None:
        output_dir = body.get("output-dir")
    if output_dir is not None:
        if not isinstance(output_dir, str) or not output_dir:
            return ApiResponse(
                status=400,
                body={"error": "output_dir must be a non-empty path"},
            )
        data["output_dir"] = output_dir

    await ctx.enqueue({"topic": "regression.buildbitstream", "data": data})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
