import os
import subprocess
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
    "name": "Build Compiler",
    "description": "build bitstream",
    "subscribes": ["compiler.build"],
    "emits": [],
    "flows": ["compiler"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    script_dir = f"{bbdir}/workflow/steps/compiler/scripts"
    yaml_dir = f"{script_dir}/yaml"
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"source {bbdir}/env.sh && mkdir -p {bbdir}/compiler/build && cd {bbdir}/compiler/build && ninja -j{os.cpu_count()}"
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        stdout_prefix="compiler build",
        stderr_prefix="compiler build",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context, result.returncode, continue_run=False
    )

    # ==================================================================================
    # Continue routing
    # ==================================================================================
    return
