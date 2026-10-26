from utils.event_common import require_target

from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design

config = {
    "name": "bebop-verilator-verilog-api",
    "description": "Generate verilog code via mill",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/verilog")],
    "enqueues": ["bebop.verilator.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    if body.get("balltype"):
        return ApiResponse(
            status=400,
            body={"error": "ball-only verilog belongs to uvm; use bbdev uvm --verilog"},
        )
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(
            status=400,
            body={
                "status": "error",
                "message": str(e),
                "example": 'bbdev bebop verilator --verilog "--design toy"',
            },
        )
    data = {"design": design, "diff": bool(body.get("diff", False))}
    await ctx.enqueue(
        {
            "topic": "bebop.verilator.verilog",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
