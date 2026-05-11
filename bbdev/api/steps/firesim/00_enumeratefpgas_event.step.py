import os
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

# Import firesim_env from scripts subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from firesim_env import setup_firesim_env

config = {
    "name": "firesim-enumeratefpgas",
    "description": "enumerate FPGAs",
    "flows": ["firesim"],
    "triggers": [queue("firesim.enumeratefpgas")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    script_dir = f"{bbdir}/bbdev/api/steps/firesim/scripts"
    yaml_dir = f"{script_dir}/yaml"

    # Setup FireSim environment variables and SSH agent
    env = setup_firesim_env()

    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"firesim enumeratefpgas"
    command += f" -a {yaml_dir}/config_hwdb.yaml"
    command += f" -b {yaml_dir}/config_build.yaml"
    command += f" -r {yaml_dir}/config_build_recipes.yaml"
    command += f" -c {yaml_dir}/config_runtime.yaml"
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        stdout_prefix="firesim enumeratefpgas",
        stderr_prefix="firesim enumeratefpgas",
        env=env,
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        ctx, result.returncode, continue_run=False, trace_id=origin_tid)

    # ==================================================================================
    # Continue routing
    # ==================================================================================

    return
