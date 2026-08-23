import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "regression-check",
    "description": "Enqueue p2e pk-tests batch for regression accuracy",
    "flows": ["regression"],
    "triggers": [queue("regression.check")],
    "enqueues": ["bebop.p2e.batch"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    chip = input_data.get("chip")
    bitstream = input_data.get("bitstream")
    if not isinstance(chip, str) or not chip or chip == "None":
        ctx.logger.error("Missing required parameter: chip")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    if not isinstance(bitstream, str) or not bitstream or bitstream == "None":
        ctx.logger.error("Missing required parameter: bitstream")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_bitstream"},
            trace_id=origin_tid,
        )
        return
    if not os.path.isfile(bitstream):
        ctx.logger.error(f"bitstream file not found: {bitstream}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "bitstream_not_found", "bitstream": bitstream},
            trace_id=origin_tid,
        )
        return

    await check_result(
        ctx,
        0,
        continue_run=True,
        extra_fields={
            "task": "regression.check",
            "chip": chip,
            "bitstream": bitstream,
        },
        trace_id=origin_tid,
    )

    data = {
        "chip": chip,
        "bitstream": bitstream,
        "test": "pk-tests",
        "from_regression_check": True,
        "_trace_id": origin_tid,
    }

    await ctx.enqueue({"topic": "bebop.p2e.batch", "data": data})
