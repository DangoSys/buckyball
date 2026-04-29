from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path
from utils.path import get_verilator_build_dir

config = {
    "name": "p2e-verilog-api",
    "description": "Generate SystemVerilog for P2E DDR4 backdoor",
    "flows": ["p2e"],
    "triggers": [api("POST", "/p2e/verilog")],
    "enqueues": ["p2e.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}

    config_name = body.get("config", "sims.p2e.P2EToyConfig")
    output_dir = get_verilator_build_dir(bbdir, config_name, body.get("output_dir"))

    data = {
        "config": config_name,
        "output_dir": output_dir,
    }
    await ctx.enqueue({"topic": "p2e.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
