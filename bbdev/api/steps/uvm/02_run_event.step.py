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
from scripts.uvm_common import run_uvm
from steps.uvm.scripts.report import coverage_report, verification_targets
from utils.reports import source_context
from utils.path import log_dir
from datetime import datetime

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
    report_context = source_context(bbdir, input_data["chip"])

    try:
        info = await asyncio.to_thread(
            run_uvm,
            bbdir,
            input_data["chip"],
            input_data.get("ball"),
            input_data.get("ip"),
            ctx,
            True,
            input_data.get("target"),
        )
        if "results" not in info:
            info = {"chip": info["chip"], "results": [info], "failures": []}
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = log_dir(bbdir, input_data["chip"], "verilog", stamp, "uvm", f"report-{origin_tid}")
        expected = verification_targets(
            bbdir, input_data["chip"], input_data.get("ball"),
            input_data.get("ip"), input_data.get("target"),
        )
        info["report_path"] = coverage_report(report_context, origin_tid, info, directory, expected)
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
        1 if info["failures"] else 0,
        continue_run=False,
        extra_fields={"task": "run", **info},
        trace_id=origin_tid,
    )
