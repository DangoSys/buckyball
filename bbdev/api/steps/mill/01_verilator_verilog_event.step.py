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
import mill as mill_run
from utils.event_common import check_result, get_origin_trace_id
from utils.stream_run import stream_run_logger_async

config = {
    "name": "verilator-verilog",
    "description": "generate chip verilog code",
    "flows": ["verilator"],
    "triggers": [queue("verilator.verilog")],
    "enqueues": ["verilator.build"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch = os.path.join(bbdir, "arch")

    try:
        chip = require_chip(input_data)
        mill_config, build_dir = rtl_out(
            bbdir, chip, "verilog", input_data.get("output_dir"),
        )
        os.makedirs(build_dir, exist_ok=True)
        ctx.logger.info(f"Using mill config: {mill_config}")
        ctx.logger.info(f"Using build directory: {build_dir}")
        returncode = (
            await stream_run_logger_async(
            cmd=mill_run.elaborate_cmd("sims.verilator.Elaborate", mill_config, build_dir),
            logger=ctx.logger,
            cwd=arch,
            stdout_prefix="verilator verilog",
            stderr_prefix="verilator verilog",
            )
        ).returncode
    except (ValueError, RuntimeError) as error:
        ctx.logger.error(str(error))
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": str(error),
                "example": 'bbdev verilator --verilog "--chip toy"',
            },
            trace_id=origin_tid,
        )
        return failure_result
    ctx.logger.info(f"Using chip: {chip}")

    await check_result(
        ctx,
        returncode,
        continue_run=input_data.get("from_run_workflow", False) and returncode == 0,
        extra_fields={"task": "verilog"},
        trace_id=origin_tid,
    )

    if input_data.get("from_run_workflow") and returncode == 0:
        await ctx.enqueue(
            {"topic": "verilator.build", "data": {**input_data, "output_dir": build_dir, "task": "run"}}
        )
