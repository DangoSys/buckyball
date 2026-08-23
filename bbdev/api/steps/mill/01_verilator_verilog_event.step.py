import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.chip import require_chip
from utils.path import get_buckyball_path
from utils.rtl import run_chip_mill
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-verilog",
    "description": "generate verilog code",
    "flows": ["verilator"],
    "triggers": [queue("verilator.verilog")],
    "enqueues": ["verilator.build"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    if input_data.get("balltype"):
        build_dir = input_data.get("output_dir")
        if not build_dir:
            ctx.logger.error("balltype mill requires output_dir")
            _, failure_result = await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={
                    "task": "validation",
                    "error": "balltype mill requires output_dir",
                },
                trace_id=origin_tid,
            )
            return failure_result
        os.makedirs(build_dir, exist_ok=True)
        command = (
            f"mill -i __.test.runMain sims.verify.BallTopMain {input_data.get('balltype')} "
            "--disable-annotation-unknown --strip-debug-info -O=release "
            f"--split-verilog -o={build_dir} "
        )
        ctx.logger.info(f"Using build directory: {build_dir}")
        result = await stream_run_logger_async(
            cmd=command,
            logger=ctx.logger,
            cwd=f"{bbdir}/arch",
            stdout_prefix="verilator verilog",
            stderr_prefix="verilator verilog",
        )
        returncode = result.returncode
    else:
        try:
            chip = require_chip(input_data)
            build_dir, returncode = await run_chip_mill(
                ctx, bbdir, chip, "verilog", "verilator verilog",
                input_data.get("output_dir"),
            )
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
