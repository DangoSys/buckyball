from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip


config = {
    "name": "vcs-verilog-api",
    "description": "generate BBSimHarness RTL for VCS",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/verilog")],
    "enqueues": ["vcs.verilog"],
}


def _request_data(body: dict) -> dict:
    chip = require_chip(body)
    data = {"chip": chip}
    if body.get("output_dir"):
        data["output_dir"] = body["output_dir"]
    return data


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    try:
        data = _request_data(req.body or {})
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})
    await ctx.enqueue({"topic": "vcs.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
