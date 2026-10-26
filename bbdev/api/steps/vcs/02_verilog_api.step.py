from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_target

config = {
    "name": "vcs-verilog-api",
    "description": "generate BBSimHarness RTL for VCS",
    "flows": ["vcs"],
    "triggers": [api("POST", "/vcs/verilog")],
    "enqueues": ["vcs.verilog"],
}


def _request_data(body: dict) -> dict:
    design = require_design(body)
    data = {"design": design, "target": require_target(body)}
    return data


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    try:
        data = _request_data(req.body or {})
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})
    await ctx.enqueue(
        {"topic": "vcs.verilog", "data": {**data, "_trace_id": ctx.trace_id}}
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
