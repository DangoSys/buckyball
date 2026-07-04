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
from scripts.uvm_common import run_uvm_build

config = {
    "name": "uvm-build",
    "description": "Build a Ball UVM simulation",
    "flows": ["uvm"],
    "triggers": [queue("uvm.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    result, info = run_uvm_build(bbdir, input_data, ctx)
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=info,
        trace_id=origin_tid,
    )
