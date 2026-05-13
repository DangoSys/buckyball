"""
bebop bemu build event handler

Builds bebop with bemu feature
"""
import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-bemu-build",
    "description": "Build bebop bemu binary",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    build_cmd = "cargo build --features bemu"
    ctx.logger.info("Building bebop bemu ...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop bemu build",
        stderr_prefix="bebop bemu build",
    )

    bebop_bin = f"{bebop_dir}/target/debug/bebop"
    await check_result(
        ctx,
        build_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "build",
            "binary": bebop_bin,
        },
        trace_id=origin_tid,
    )
