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
    "name": "Marshal Launch",
    "description": "launch marshal",
    "subscribes": ["marshal.launch"],
    "emits": [],
    "flows": ["marshal"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    script_dir = f"{bbdir}/workflow/steps/marshal/scripts"
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"source {bbdir}/env.sh && ./marshal -v launch interactive.json"
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=script_dir,
        stdout_prefix="marshal launch",
        stderr_prefix="marshal launch",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context, result.returncode, continue_run=False
    )

    # ==================================================================================
    # Continue routing
    # Finish workflow
    # ==================================================================================
    return
