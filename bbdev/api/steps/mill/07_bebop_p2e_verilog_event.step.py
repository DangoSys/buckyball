import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.chip import require_chip
from utils.path import get_buckyball_path
from utils.rtl import run_chip_mill
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-p2e-verilog",
    "description": "Generate P2E Verilog via mill",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.verilog")],
    "enqueues": ["bebop.p2e.buildbitstream"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    try:
        chip = require_chip(input_data)
        build_dir, returncode = await run_chip_mill(
            ctx, bbdir, chip, "p2e", "bebop p2e verilog",
            input_data.get("output_dir"),
        )
    except (ValueError, RuntimeError) as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip", "detail": str(error)},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"Using P2E chip: {chip}")
    ctx.logger.info(f"Using P2E output directory: {build_dir}")

    from_reg = bool(input_data.get("from_regression_buildbitstream"))
    await check_result(
        ctx,
        returncode,
        continue_run=from_reg and returncode == 0,
        extra_fields={
            "task": "verilog",
            "chip": chip,
            "output_dir": build_dir,
            "top_module": "P2ETop",
        },
        trace_id=origin_tid,
    )
    if from_reg and returncode == 0:
        await ctx.enqueue(
            {
                "topic": "bebop.p2e.buildbitstream",
                "data": {
                    **input_data,
                    "vsrc_dir": build_dir,
                    "from_regression_buildbitstream": True,
                    "_trace_id": origin_tid,
                },
            }
        )
