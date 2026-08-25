from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "uvm-ball-verilog-api",
    "description": "generate isolated ball verilog for UVM",
    "flows": ["uvm"],
    "triggers": [api("POST", "/uvm/verilog")],
    "enqueues": ["uvm.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    if body.get("chip") or (
        isinstance(body.get("config"), str) and body.get("config") and body.get("config") != "None"
    ):
        return ApiResponse(
            status=400,
            body={"error": "uvm ball verilog does not take --chip or --config"},
        )

    balltype = body.get("balltype")
    if not isinstance(balltype, str) or not balltype:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --balltype"},
        )

    output_dir = body.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --output-dir"},
        )

    data = {"balltype": balltype, "output_dir": output_dir}
    await ctx.enqueue({"topic": "uvm.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
