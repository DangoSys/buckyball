from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.chip import require_chip

config = {
    "name": "verilator-verilog-api",
    "description": "generate verilog code",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/verilog")],
    "enqueues": ["verilator.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    if body.get("balltype"):
        if body.get("chip") or (
            isinstance(body.get("config"), str) and body.get("config") and body.get("config") != "None"
        ):
            return ApiResponse(
                status=400,
                body={"error": "balltype mill does not take --chip or --config"},
            )
        output_dir = body.get("output_dir")
        if not output_dir:
            return ApiResponse(
                status=400,
                body={"error": "balltype mill requires output_dir"},
            )
        data = {
            "balltype": body.get("balltype"),
            "output_dir": output_dir,
        }
    else:
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
