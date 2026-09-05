import asyncio
import os
import sys

from motia import FlowContext, queue

step_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.abspath(os.path.join(step_dir, "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from scripts.uvm_common import run_chip

config = {
    "name": "uvm-run",
    "description": "Build and run a Ball UVM simulation",
    "flows": ["uvm"],
    "triggers": [queue("uvm.run")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    try:
        info = await asyncio.to_thread(
            run_chip, bbdir, input_data["chip"], input_data.get("ball"), ctx, True
        )
    except Exception as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": str(e)},
            trace_id=origin_tid,
        )
        return

    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={"task": "run", **info},
        trace_id=origin_tid,
    )
