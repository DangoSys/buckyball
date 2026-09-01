import os
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import check_result, get_origin_trace_id, require_chip
from utils.path import get_buckyball_path, rtl_dir
from generate import generate_sram


config = {
    "name": "ip-generate",
    "description": "generate SRAM macros from elaborator mems.conf via MacroCompiler",
    "flows": ["ip", "dc", "yosys"],
    "triggers": [queue("ip.generate")],
    "enqueues": ["ip.replace"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
        consumer = input_data.get("consumer") or "dc"
        if consumer not in ("dc", "yosys"):
            raise ValueError("consumer must be dc or yosys")
        top = input_data.get("top") or "DigitalTop"
        if not isinstance(top, str) or not top:
            raise ValueError("top must be a non-empty string")
        bbdir = Path(get_buckyball_path()).resolve()
        build_dir = Path(rtl_dir(bbdir, chip, "verilog", input_data.get("output_dir"))).resolve()
        man = generate_sram(bbdir=bbdir, chip=chip, build_dir=build_dir)
    except Exception as exc:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "ip.generate", "error": str(exc)},
            trace_id=origin_tid,
        )
        return failure_result

    extra = {
        "task": "ip.generate",
        "chip": man["chip"],
        "generate_manifest": man["generate_manifest"],
        "sram_macros_v": man["sram_macros_v"],
        "mem_count": man["mem_count"],
        "build_dir": str(build_dir),
    }
    await check_result(ctx, 0, continue_run=True, extra_fields=extra, trace_id=origin_tid)
    await ctx.enqueue(
        {
            "topic": "ip.replace",
            "data": {
                **input_data,
                "chip": chip,
                "consumer": consumer,
                "top": top,
                "_trace_id": origin_tid,
            },
        }
    )
