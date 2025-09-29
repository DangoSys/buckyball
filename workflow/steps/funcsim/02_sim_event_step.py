from contextlib import redirect_stdout
import os
from re import T
import subprocess
import sys
import time
from datetime import datetime

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)


from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.search_workload import search_workload
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "functional simulation",
    "description": "functional simulation",
    "subscribes": ["funcsim.sim"],
    "emits": [],
    "flows": ["funcsim"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    extension = data.get("ext", "")

    binary_name = data.get("binary", "")
    binary_path = search_workload(f"{bbdir}/bb-tests/output/workloads/src", binary_name)
    funcsim_dir = f"{bbdir}/sims/func-sim"

    output_dir = f"{bbdir}/workflow/steps/funcsim/output"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = f"{output_dir}/{timestamp}-{binary_name}"

    os.makedirs(log_dir, exist_ok=True)

    command = f"source {bbdir}/env.sh && cd {funcsim_dir} && spike --extension={extension} -l --ic=64:4:64 --dc=64:4:64 --l2=256:8:128 --log-cache-miss {binary_path} 2> {log_dir}/stderr.log | tee {log_dir}/stdout.log"
    context.logger.info(
        "Executing funcsim command", {"command": command, "cwd": funcsim_dir}
    )
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=funcsim_dir,
        executable="bash",
        stdout_prefix="funcsim sim",
        stderr_prefix="funcsim sim",
    )

    # ==================================================================================
    # 返回仿真结果
    # ==================================================================================
    # 此处为run workflow的终点，status状态不再继续设为processing
    success_result, failure_result = await check_result(
        context, result.returncode, continue_run=False
    )

    # ==================================================================================
    #  finish workflow
    # ==================================================================================
    return
