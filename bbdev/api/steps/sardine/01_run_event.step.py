import os
import sys
import time

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "sardine-run",
    "description": "running sardine",
    "flows": ["sardine"],
    "triggers": [queue("sardine.run")],
    "enqueues": ["sardine.coverage_report"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    sardine_dir = f"{bbdir}/bb-tests/sardine"

    # Record start time so coverage report can filter to only this run's .dat files
    run_start_time = time.time()

    command = f"python3 run_tests.py -m \"({input_data.get('workload', '')})\""
    if input_data.get("coverage", False):
        command += " --coverage"
    ctx.logger.info(
        "Executing sardine command", {"command": command, "cwd": sardine_dir}
    )
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=sardine_dir,
        executable="bash",
        stdout_prefix="sardine run",
        stderr_prefix="sardine run",
    )

    # ==================================================================================
    # Return execution result
    # ==================================================================================
    if input_data.get("coverage", False):
        # When coverage is enabled, always continue to coverage report step
        # (even if some tests failed, coverage data is still valid)
        await check_result(ctx, result.returncode, continue_run=True, trace_id=origin_tid)
        await ctx.enqueue(
            {"topic": "sardine.coverage_report", "data": {**input_data, "run_start_time": run_start_time}}
        )
    else:
        await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)

    return
