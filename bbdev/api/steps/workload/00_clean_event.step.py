import os
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path

config = {
    "name": "workload-clean",
    "description": "clean workload output directory",
    "flows": ["workload"],
    "triggers": [queue("workload.clean")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    allowed = {"_trace_id"}
    unknown = sorted(k for k in input_data if k not in allowed)
    if unknown:
        ctx.logger.error(f"Unknown workload clean parameter(s): {', '.join(unknown)}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "unknown_parameter", "parameters": unknown},
            trace_id=origin_tid,
        )
        return

    bbdir = get_buckyball_path()
    paths = [os.path.join(bbdir, "bb-tests", "output")]

    for path in paths:
        if os.path.exists(path):
            ctx.logger.info("Removing workload directory", {"path": path})
            shutil.rmtree(path)
        else:
            ctx.logger.info("Workload directory already clean", {"path": path})

    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={"task": "clean", "paths": paths},
        trace_id=origin_tid,
    )
