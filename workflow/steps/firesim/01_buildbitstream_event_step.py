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
    "name": "Firesim Buildbitstream",
    "description": "build bitstream",
    "subscribes": ["firesim.buildbitstream"],
    "emits": [],
    "flows": ["firesim"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    script_dir = f"{bbdir}/workflow/steps/firesim/scripts"
    yaml_dir = f"{script_dir}/yaml"
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"source {bbdir}/env.sh && firesim buildbitstream "
    command += f" -a {yaml_dir}/config_hwdb.yaml"
    command += f" -b {yaml_dir}/config_build.yaml"
    command += f" -r {yaml_dir}/config_build_recipes.yaml"
    command += f" -c {yaml_dir}/config_runtime.yaml"
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        stdout_prefix="firesim buildbitstream",
        stderr_prefix="firesim buildbitstream",
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
