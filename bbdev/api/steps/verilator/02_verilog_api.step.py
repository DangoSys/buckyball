from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "verilator-verilog-api",
    "description": "generate verilog code",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/verilog")],
    "enqueues": ["verilator.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    # Get config name, must be provided
    config_name = body.get("config")
    if not config_name or config_name == "None":
        return ApiResponse(
            status=400,
            body={
                "status": "error",
                "message": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
        )

    data = {
        "config": config_name,
        "balltype": body.get("balltype"),
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
    }
    await ctx.enqueue({"topic": "verilator.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
