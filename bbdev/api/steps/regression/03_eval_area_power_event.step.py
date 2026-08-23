import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.chip import require_chip
from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path

config = {
    "name": "regression-eval-area-power",
    "description": "Enqueue DC area workflow for regression area+freq",
    "flows": ["regression"],
    "triggers": [queue("regression.eval-area-power")],
    "enqueues": ["dc.verilog"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    try:
        chip = require_chip(input_data)
    except ValueError as exc:
        ctx.logger.error(str(exc))
        extra = {"error": "missing_chip"} if str(exc).startswith("Missing required parameter: --chip") else {"error": str(exc)}
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields=extra,
            trace_id=origin_tid,
        )
        return
    tapeout = os.path.join(bbdir, "examples", "chips", chip, "tapeout")
    if not os.path.isdir(tapeout):
        ctx.logger.error(f"chip {chip} has no tapeout directory: {tapeout}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_tapeout", "chip": chip, "tapeout": tapeout},
            trace_id=origin_tid,
        )
        return

    top = input_data.get("top") or "DigitalTop"

    await check_result(
        ctx,
        0,
        continue_run=True,
        extra_fields={
            "task": "regression.eval-area-power",
            "chip": chip,
        },
        trace_id=origin_tid,
    )

    await ctx.enqueue(
        {
            "topic": "dc.verilog",
            "data": {
                "chip": chip,
                "top": top,
                "from_area_workflow": True,
                "from_regression_area_power": True,
                "_trace_id": origin_tid,
            },
        }
    )
