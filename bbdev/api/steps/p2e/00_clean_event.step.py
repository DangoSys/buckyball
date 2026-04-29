import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "p2e-clean",
    "description": "Clean P2E build directory",
    "flows": ["p2e"],
    "triggers": [queue("p2e.clean")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    config_name = input_data.get("config", "sims.p2e.P2EToyConfig")
    build_dir = get_verilator_build_dir(bbdir, config_name, input_data.get("output_dir"))

    command = f"rm -rf {build_dir}"
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="p2e clean",
        stderr_prefix="p2e clean",
    )

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"task": "clean", "config": config_name, "output_dir": build_dir},
        trace_id=origin_tid,
    )

