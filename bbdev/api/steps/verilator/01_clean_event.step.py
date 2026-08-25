import os
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-clean",
    "description": "clean build directory",
    "flows": ["verilator"],
    "triggers": [
        queue("verilator.run"),
        queue("verilator.clean"),
    ],
    "enqueues": ["verilator.verilog"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    bbdir = get_buckyball_path()
    build_dir = rtl_dir(
        bbdir, chip,
        "verilog",
        input_data.get("output_dir"),
    )
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"rm -rf {build_dir}"
    result = await stream_run_logger_async(
        cmd=command,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="verilator clean",
        stderr_prefix="verilator clean",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "clean"}, trace_id=origin_tid,
    )

    # ==================================================================================
    # Continue routing
    # ==================================================================================
    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {"topic": "verilator.verilog", "data": {**input_data, "output_dir": build_dir, "task": "run"}}
        )

    return
