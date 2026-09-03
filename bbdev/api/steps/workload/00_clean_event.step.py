import os
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.event_common import check_result, get_origin_trace_id
from utils.path import chip_output_root, get_buckyball_path

config = {
    "name": "workload-clean",
    "description": "clean workload output directory for one chip",
    "flows": ["workload"],
    "triggers": [queue("workload.clean")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    allowed = {"chip", "_trace_id"}
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

    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return

    bbdir = get_buckyball_path()
    output_path = chip_output_root(bbdir, chip)
    build_path = os.path.join(bbdir, "bb-tests", "workloads", "build", chip)

    for path in (output_path, build_path):
        if os.path.exists(path):
            ctx.logger.info("Removing workload directory", {"path": path})
            shutil.rmtree(path)
        else:
            ctx.logger.info("Workload directory already clean", {"path": path})

    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "task": "clean",
            "chip": chip,
            "output_path": output_path,
            "build_path": build_path,
        },
        trace_id=origin_tid,
    )
