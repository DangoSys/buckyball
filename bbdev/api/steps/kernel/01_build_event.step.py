import os
import sys
import shutil

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

# Import bin_to_hex converter
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
from bin_to_hex import bin_to_hex

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
    output_dir = os.path.join(bbdir, "bb-tests", "output", "kernel")

    # Clear previous output before rebuild
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

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

    if result.returncode != 0:
        await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)
        return

    # Convert bin to hex for P2E memory backdoor
    bin_file = os.path.join(output_dir, "pegasus-bin")
    hex_file = os.path.join(output_dir, "pegasus.hex")

    if os.path.exists(bin_file):
        ctx.logger.info(f"Converting {bin_file} to Verilog hex format...")
        success = bin_to_hex(bin_file, hex_file)
        if not success:
            ctx.logger.warning("Failed to convert bin to hex, but continuing...")

    await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)

