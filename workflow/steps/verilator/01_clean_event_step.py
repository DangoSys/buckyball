import subprocess
import os
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "make clean",
    "description": "clean build directory",
    "subscribes": ["verilator.run", "verilator.clean"],
    "emits": ["verilator.verilog"],
    "flows": ["verilator"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    build_dir = f"{bbdir}/arch/build"
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"rm -rf {build_dir}"
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="verilator clean",
        stderr_prefix="verilator clean",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=data.get("from_run_workflow", False),
        extra_fields={"task": "clean"},
    )

    # ==================================================================================
    # Continue routing
    # ==================================================================================
    if data.get("from_run_workflow"):
        await context.emit(
            {"topic": "verilator.verilog", "data": {**data, "task": "run"}}
        )

    return
