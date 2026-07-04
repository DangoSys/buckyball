import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from yosys_log import make_yosys_log_dir, req_arg

config = {
    "name": "yosys-run",
    "description": "run yosys flow",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/run")],
    "enqueues": ["yosys.run"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}
    log_dir = req_arg(body, "log_dir") or make_yosys_log_dir(bbdir, ctx.trace_id)

    data = {
        "output_dir": req_arg(body, "output_dir") or f"{bbdir}/arch/build/",
        "log_dir": log_dir,
        "top": req_arg(body, "top"),
        "config": req_arg(body, "config"),
        "from_run_workflow": True,
    }
    await ctx.enqueue({"topic": "yosys.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
