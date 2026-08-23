import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import get_buckyball_path
from utils.event_common import check_result, get_origin_trace_id
from bemu_analysis import abs_log_dir, analysis_dir, chip_maps, write_report

config = {
    "name": "bebop-bemu-analysis",
    "description": "Analyze bebop bemu itrace/mtrace",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.analysis")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return

    try:
        log_path = abs_log_dir(input_data.get("log-dir") or input_data.get("log_dir"))
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "log_dir_not_absolute", "message": str(e)},
            trace_id=origin_tid,
        )
        return

    itrace = bool(input_data.get("itrace", False))
    mtrace = bool(input_data.get("mtrace", False))
    try:
        names, matrix, depth = chip_maps(get_buckyball_path(), chip)
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
            ctx, 1, continue_run=False,
            extra_fields={"error": "analysis_failed", "message": str(e), "log_dir": str(log_path)},
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
            "chip": chip,
            "log_dir": str(log_path),
            "analysis": str(out),
        },
        trace_id=origin_tid,
    )
