import asyncio
import importlib.util
import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

_stack_build_path = os.path.join(os.path.dirname(__file__), "scripts", "build.py")
_spec = importlib.util.spec_from_file_location("stack_build_module", _stack_build_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_stack_build_path}")
stack_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stack_build)

from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "build-stack",
    "description": "Build one level of a design's stack",
    "flows": ["stack"],
    "triggers": [queue("stack.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    log = {"logger": ctx.logger, "task_scope": origin_tid}
    try:
        await asyncio.to_thread(
            stack_build.build_design,
            bbdir,
            design=input_data["design"],
            selection=input_data["target"],
            **log,
        )
    except Exception as exc:
        ctx.logger.error(f"Stack build failed: {exc}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "build", "error": str(exc)},
            trace_id=origin_tid,
        )
        return
    await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)
