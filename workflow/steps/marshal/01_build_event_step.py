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
    "name": "Marshal Build",
    "description": "build marshal",
    "subscribes": ["marshal.build"],
    "emits": [],
    "flows": ["marshal"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    script_dir = f"{bbdir}/workflow/steps/marshal/scripts"
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"source {bbdir}/env.sh && ./marshal -v build interactive.json  && ./marshal -v install interactive.json"
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=script_dir,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context, result.returncode, continue_run=False
    )

    # ==================================================================================
    # Continue routing
    # Routing to completion or error handling
    # ==================================================================================
    if result.returncode == 0:
        await context.emit(
            {
                "topic": "marshal.complete",
                "data": {**data, "task": "marshal", "result": success_result},
            }
        )
    else:
        await context.emit(
            {
                "topic": "marshal.error",
                "data": {
                    **data,
                    "task": "marshal",
                    "result": failure_result,
                    "returncode": result.returncode,
                },
            }
        )

    return
