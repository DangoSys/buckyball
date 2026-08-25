import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import rtl_dir, get_buckyball_path

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from yosys_log import req_arg

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
    try:
        chip = require_chip(body)
        rtl = rtl_dir(bbdir, chip, "synth", req_arg(body, "output_dir"))
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {
        "chip": chip,
        "output_dir": rtl,
        "top": req_arg(body, "top") or "DigitalTop",
        "vcd": req_arg(body, "vcd"),
    }
    log_dir = req_arg(body, "log_dir")
    if log_dir:
        data["log_dir"] = log_dir
    await ctx.enqueue(
        {
            "topic": "ip-replace.run",
            "data": {
                **data,
                "source_list": os.path.join(rtl, "yosys_sources.list"),
                "ip_replace_output_dir": log_dir or rtl,
                "consumer": "yosys",
                "next_topic": "yosys.synth",
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
