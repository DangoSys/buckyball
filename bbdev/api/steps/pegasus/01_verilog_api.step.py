from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "Pegasus Verilog",
    "description": "Generate SystemVerilog for Pegasus FPGA (PegasusHarness + ChipTop)",
    "flows": ["pegasus"],
    "triggers": [api("POST", "/pegasus/verilog")],
    "enqueues": ["pegasus.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}

    config_name = body.get("config", "sims.pegasus.PegasusConfig")

    data = {
        "config": config_name,
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/pegasus/"),
    }
    await ctx.enqueue({"topic": "pegasus.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
