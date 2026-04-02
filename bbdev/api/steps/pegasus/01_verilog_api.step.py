from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path
from utils.event_common import wait_for_result

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

    # Default config for Pegasus; allow override
    config_name = body.get("config", "sims.pegasus.PegasusConfig")

    data = {
        "config": config_name,
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/pegasus/"),
    }

    await ctx.enqueue({"topic": "pegasus.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    result = await wait_for_result(ctx)
    return result
