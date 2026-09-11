import asyncio
import os
import sys
from pathlib import Path

from motia import FlowContext, queue

step_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.abspath(os.path.join(step_dir, "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)

from scripts.waive import apply_waivers
from utils.event_common import check_result, get_origin_trace_id, require_chip
from utils.path import get_buckyball_path, rtl_dir

config = {
    "name": "uvm-waive",
    "description": "apply common Chisel coverage waivers to generated RTL",
    "flows": ["uvm"],
    "triggers": [queue("uvm.waive")],
    "enqueues": ["uvm.build", "uvm.run"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
        task = input_data["uvm_task"]
        if task not in ("build", "run"):
            raise ValueError(f"invalid UVM task: {task}")
        bbdir = get_buckyball_path()
        count = await asyncio.to_thread(apply_waivers, Path(rtl_dir(bbdir, chip, "tapeout")))
    except (KeyError, ValueError) as error:
        await check_result(ctx, 1, extra_fields={"task": "waive", "error": str(error)}, trace_id=origin_tid)
        return
    await check_result(ctx, 0, continue_run=True, extra_fields={"task": "waive", "count": count}, trace_id=origin_tid)
    await ctx.enqueue(
        {"topic": f"uvm.{task}", "data": {**input_data, "force_rebuild": True, "_trace_id": origin_tid}}
    )
