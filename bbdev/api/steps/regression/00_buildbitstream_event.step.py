import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "regression-buildbitstream",
    "description": "Enqueue p2e verilog then buildbitstream for regression",
    "flows": ["regression"],
    "triggers": [queue("regression.buildbitstream")],
    "enqueues": ["bebop.p2e.verilog"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
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

    await check_result(
        ctx,
        0,
        continue_run=True,
        extra_fields={
            "task": "regression.buildbitstream",
            "chip": chip,
        },
        trace_id=origin_tid,
    )

    data = {
        "chip": chip,
        "from_regression_buildbitstream": True,
        "_trace_id": origin_tid,
    }
    output_dir = input_data.get("output_dir") or input_data.get("output-dir")
    if output_dir:
        data["build_dir"] = output_dir

    await ctx.enqueue({"topic": "bebop.p2e.verilog", "data": data})
