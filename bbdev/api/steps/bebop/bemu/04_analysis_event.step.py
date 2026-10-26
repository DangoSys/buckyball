import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import get_buckyball_path, run_log_dir
from utils.event_common import check_result, get_origin_trace_id
from bemu_analysis import analysis_dir, design_maps, write_report

config = {
    "name": "bebop-bemu-analysis",
    "description": "Analyze bebop bemu itrace/mtrace",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.analysis")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    design = input_data.get("design")
    if not design:
        ctx.logger.error("Missing required parameter: design")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_design"},
            trace_id=origin_tid,
        )
        return

    try:
        log_path = run_log_dir(get_buckyball_path(), input_data.get("run"))
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_run", "message": str(e)},
            trace_id=origin_tid,
        )
        return

    itrace = bool(input_data.get("itrace", False))
    mtrace = bool(input_data.get("mtrace", False))
    try:
        names, matrix, depth = design_maps(get_buckyball_path(), design)
        text = analysis_dir(
            log_path,
            names=names,
            matrix=matrix,
            bank_depth=depth,
            itrace=itrace,
            mtrace=mtrace,
        )
        out = write_report(log_path, text)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "error": "analysis_failed",
                "message": str(e),
                "log_dir": str(log_path),
            },
            trace_id=origin_tid,
        )
        return

    for line in text.splitlines():
        ctx.logger.info(line)
    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "task": "bemu-analysis",
            "design": design,
            "log_dir": str(log_path),
            "analysis": str(out),
        },
        trace_id=origin_tid,
    )
