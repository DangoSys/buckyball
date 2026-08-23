import os
import re
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.chip import available_compiler_chips, available_cores, resolve_core
from utils.build import build_compiler
from utils.path import get_buckyball_path
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

    stable = input_data.get("stable", False)
    if not isinstance(stable, bool):
        ctx.logger.error("Invalid parameter: stable must be a boolean flag")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_stable", "stable": stable},
            trace_id=origin_tid,
        )
        return

    chip = input_data.get("chip")
    core = input_data.get("core")
    if bool(chip) == bool(core):
        ctx.logger.error("Specify exactly one compiler target: chip or core")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_target_selection"},
            trace_id=origin_tid,
        )
        return
    target_name = core or chip
    if not isinstance(target_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", target_name):
        ctx.logger.error(f"Invalid compiler target: {target_name}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_target", "target": target_name},
            trace_id=origin_tid,
        )
        return

    try:
        if chip:
            build_compiler(bbdir, chip=chip)
        else:
            resolve_core(bbdir, core, require_compiler=True)
            build_compiler(bbdir, core=core)
    except (ValueError, RuntimeError) as error:
        choices = available_cores(bbdir) if core else available_compiler_chips(bbdir)
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "unknown_core" if core else "unknown_chip",
                "core": core,
                "chip": chip,
                "available": choices,
            },
            trace_id=origin_tid,
        )
        return

    await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)
