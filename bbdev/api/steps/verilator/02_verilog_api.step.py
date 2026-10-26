from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design

config = {
    "name": "verilator-verilog-api",
    "description": "generate design verilog code",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/verilog")],
    "enqueues": ["verilator.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})

    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(
            status=400,
            body={
                "status": "error",
                "message": str(e),
                "example": 'bbdev verilator --verilog "--design toy"',
            },
        )

    data = {"design": design}
    await ctx.enqueue(
        {
            "topic": "verilator.verilog",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
