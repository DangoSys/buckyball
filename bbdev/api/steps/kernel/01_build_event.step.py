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
    "description": "build RISC-V kernel + rootfs for image via bb-tests/workloads/lib/kernel",
    "flows": ["kernel"],
    "triggers": [queue("kernel.build")],
    "enqueues": [],
}


def hart_count_params(input_data: dict) -> dict:
    visible = int(input_data.get("visible-hart-count", 64))
    total = int(input_data.get("total-hart-count", visible))
    hidden_base = int(input_data.get("hidden-hart-base", visible))

    if visible < 1:
        raise ValueError("visible-hart-count must be at least 1")
    if total < visible:
        raise ValueError("total-hart-count must cover visible harts")
    if hidden_base < visible:
        raise ValueError("hidden-hart-base must be after visible harts")

    return {
        "visible": visible,
        "total": total,
        "hidden_base": hidden_base,
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

    try:
        hart_params = hart_count_params(input_data)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    # cmake configure
    configure_cmd = (
        f"cmake -B {kernel_build} -S {kernel_src} "
        f"-DBUCKYBALL_VISIBLE_HART_COUNT={hart_params['visible']} "
        f"-DBUCKYBALL_TOTAL_HART_COUNT={hart_params['total']} "
        f"-DBUCKYBALL_HIDDEN_HART_BASE={hart_params['hidden_base']}"
    )
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

    # Convert fw_payload.bin to hex for P2E memory backdoor
    fw_payload_bin = os.path.join(output_dir, "fw_payload.bin")
    fw_payload_hex = os.path.join(output_dir, "fw_payload.hex")

    if not os.path.exists(fw_payload_bin):
        ctx.logger.error("fw_payload.bin not found")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    ctx.logger.info(f"Converting {fw_payload_bin} to Verilog hex format for P2E...")
    success = bin_to_hex(fw_payload_bin, fw_payload_hex, base_address=0x80000000)
    if not success:
        ctx.logger.error("Failed to convert fw_payload to hex")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)
