import os
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
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
    bbdir = get_buckyball_path()
    build_dir = get_verilator_build_dir(
        bbdir,
        input_data.get("config"),
        input_data.get("output_dir"),
    )
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"rm -rf {build_dir}"
    result = stream_run_logger(
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
