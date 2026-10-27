import asyncio
import os
import sys

from motia import FlowContext, queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.event_common import check_result, get_origin_trace_id
from steps.model.scripts.build import build_model
from utils.path import get_buckyball_path

config = {
    "name": "model-build",
    "description": "Build a design model recipe",
    "flows": ["model"],
    "triggers": [queue("model.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    trace_id = get_origin_trace_id(input_data, ctx)
    params = {key: value for key, value in input_data.items() if key != "_trace_id"}
    try:
        await asyncio.to_thread(
            build_model,
            get_buckyball_path(),
            params["design"],
            params["model"],
            selection=params["target"],
            logger=ctx.logger,
            task_scope=trace_id,
        )
    except Exception as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "model_build_failed", "detail": str(error)},
            trace_id=trace_id,
        )
        return
    await check_result(
        ctx, 0, continue_run=False, extra_fields=params, trace_id=trace_id
    )
