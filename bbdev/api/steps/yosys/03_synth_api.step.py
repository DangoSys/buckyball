import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from yosys_log import make_yosys_log_dir, req_arg

config = {
    "name": "yosys-synth-api",
    "description": "run yosys synthesis for area estimation",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/synth")],
    "enqueues": ["ip-replace.run"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}
    log_dir = req_arg(body, "log_dir") or make_yosys_log_dir(bbdir, ctx.trace_id)

    data = {
        "output_dir": req_arg(body, "output_dir") or f"{bbdir}/arch/build/",
        "log_dir": log_dir,
        "top": req_arg(body, "top") or "DigitalTop",
        "config": req_arg(body, "config"),
        "vcd": req_arg(body, "vcd"),
    }
    await ctx.enqueue(
        {
            "topic": "ip-replace.run",
            "data": {
                **data,
                "source_list": f"{data['output_dir']}/yosys_sources.list",
                "ip_replace_output_dir": log_dir,
                "consumer": "yosys",
                "next_topic": "yosys.synth",
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
