import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "kernel-build",
    "description": "build RISC-V kernel + rootfs for Pegasus via bb-tests/workloads/lib/kernel",
    "flows": ["kernel"],
    "triggers": [queue("kernel.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    kernel_src = os.path.join(bbdir, "bb-tests", "workloads", "lib", "kernel")
    kernel_build = os.path.join(bbdir, "bb-tests", "build", "kernel")

    # cmake configure
    configure_cmd = f"cmake -B {kernel_build} -S {kernel_src}"
    result = stream_run_logger(
        cmd=configure_cmd,
        logger=ctx.logger,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )
    if result.returncode != 0:
        await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)
        return

    # cmake build
    build_cmd = f"cmake --build {kernel_build} --target kernel-build"
    result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )

    await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)

