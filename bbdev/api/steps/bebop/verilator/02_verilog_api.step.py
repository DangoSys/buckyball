from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-verilator-verilog-api",
    "description": "Generate verilog code via mill",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/verilog")],
    "enqueues": ["bebop.verilator.verilog"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    config_name = body.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        return ApiResponse(
            status=400,
            body={
                "status": "error",
                "message": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev bebop verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
        )

    data = {
        "config": config_name,
        "balltype": body.get("balltype"),
        "output_dir": get_verilator_build_dir(bbdir, config_name, body.get("output_dir")),
    }
    await ctx.enqueue({"topic": "bebop.verilator.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
