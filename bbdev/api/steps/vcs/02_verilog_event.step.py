import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.chip import require_chip
from utils.path import get_buckyball_path
from utils.rtl import run_chip_mill


config = {
    "name": "vcs-verilog",
    "description": "generate BBSimHarness RTL for VCS",
    "flows": ["vcs"],
    "triggers": [queue("vcs.verilog")],
    "enqueues": ["vcs.build"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
    except ValueError as exc:
        await check_result(ctx, 1, extra_fields={"task": "verilog", "error": str(exc)}, trace_id=origin_tid)
        return
    bbdir = get_buckyball_path()
    try:
        build_dir, returncode = await run_chip_mill(
            ctx, bbdir, chip, "verilog", "vcs verilog",
            input_data.get("output_dir"),
        )
    except (ValueError, RuntimeError) as exc:
        await check_result(ctx, 1, extra_fields={"task": "verilog", "error": str(exc)}, trace_id=origin_tid)
        return
    await check_result(
        ctx,
        returncode,
        continue_run=input_data.get("from_run_workflow", False) and returncode == 0,
        extra_fields={"task": "verilog", "output_dir": build_dir},
        trace_id=origin_tid,
    )
    if input_data.get("from_run_workflow") and returncode == 0:
        await ctx.enqueue({"topic": "vcs.build", "data": {**input_data, "output_dir": build_dir, "_trace_id": origin_tid}})
