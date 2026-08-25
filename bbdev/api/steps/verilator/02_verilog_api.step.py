from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip

config = {
    "name": "verilator-verilog-api",
    "description": "generate chip verilog code",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/verilog")],
    "enqueues": ["verilator.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    if body.get("balltype"):
        return ApiResponse(
            status=400,
            body={
                "error": "ball-only verilog belongs to uvm; use bbdev uvm --verilog",
            },
        )

    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(
            status=400,
            body={
                "status": "error",
                "message": str(e),
                "example": 'bbdev verilator --verilog "--chip toy"',
            },
        )

    data = {"chip": chip}
    if body.get("output_dir"):
        data["output_dir"] = body["output_dir"]
    await ctx.enqueue({"topic": "verilator.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
