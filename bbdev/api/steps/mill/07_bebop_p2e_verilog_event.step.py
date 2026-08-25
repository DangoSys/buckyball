import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
mill_dir = os.path.dirname(__file__)
mill_scripts = os.path.join(mill_dir, "scripts")
if mill_dir not in sys.path:
    sys.path.insert(0, mill_dir)
if mill_scripts not in sys.path:
    sys.path.insert(0, mill_scripts)

from utils.event_common import require_chip
from utils.path import get_buckyball_path
from utils.path import rtl_out
from utils.stream_run import stream_run_logger_async
import mill as mill_run
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
    arch = os.path.join(bbdir, "arch")
    try:
        chip = require_chip(input_data)
        mill_config, build_dir = rtl_out(
            bbdir, chip, "p2e", input_data.get("output_dir"),
        )
        os.makedirs(build_dir, exist_ok=True)
        ctx.logger.info(f"Using mill config: {mill_config}")
        ctx.logger.info(f"Using build directory: {build_dir}")
        prefix = "bebop p2e verilog"
        returncode = (
            await stream_run_logger_async(
            cmd=mill_run.elaborate_cmd("sims.p2e.Elaborate", mill_config, build_dir),
            logger=ctx.logger,
            cwd=arch,
            stdout_prefix=prefix,
            stderr_prefix=prefix,
            )
        ).returncode
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
