import asyncio
import os
import re
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import build as compiler_build  # noqa: E402

from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "build-compiler",
    "description": "build compiler",
    "flows": ["compiler"],
    "triggers": [queue("compiler.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    chip = input_data.get("chip")
    if not isinstance(chip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        raise ValueError(f"Invalid compiler chip: {chip}")

    try:
        await asyncio.to_thread(
            compiler_build.build_compiler,
            bbdir,
            chip=chip,
            logger=ctx.logger,
            task_scope=origin_tid,
        )
    except Exception as exc:
        ctx.logger.error(f"Compiler build failed: {exc}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "build", "chip": chip, "error": str(exc)},
            trace_id=origin_tid,
        )
        return

    await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)
