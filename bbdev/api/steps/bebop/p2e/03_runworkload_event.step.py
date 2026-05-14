"""
bebop p2e runworkload event handler

Loads a kernel image into FPGA and runs the workload via bebop CLI:
  1. Resolve image name to .hex file under bb-tests/output/
  2. Validate bitstream .bit file path
  3. Run bebop p2e --runworkload --image <image-path> --bitstream <bitstream> --log-dir <log>
"""
import glob
import os
import sys
from datetime import datetime

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-p2e-runworkload",
    "description": "Run workload on FPGA via bebop p2e CLI",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.runworkload")],
    "enqueues": [],
}


def resolve_image(bbdir: str, image_name: str) -> str:
    """Search bb-tests/output/ recursively for <image_name>.hex and return its absolute path."""
    output_root = f"{bbdir}/bb-tests/output"
    matches = glob.glob(f"{output_root}/**/{image_name}.hex", recursive=True)
    if not matches:
        return ""
    return matches[0]


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    image_name = input_data.get("image", "")
    bitstream = input_data.get("bitstream", "")

    image_path = resolve_image(bbdir, image_name)
    if not image_path:
        ctx.logger.error(f"image .hex not found for name: {image_name} (searched bb-tests/output/)")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "image_not_found", "image": image_name},
            trace_id=origin_tid,
        )
        return

    if not bitstream or not os.path.isfile(bitstream):
        ctx.logger.error(f"bitstream .bit file not found: {bitstream}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "bitstream_not_found", "bitstream": bitstream},
            trace_id=origin_tid,
        )
        return

    # Infer build_dir from bitstream path
    # bitstream is typically: <build_dir>/fpgaCompDir/part_b0_f0/pnrDir/xepic_vvac_top_0_0.bit
    # so build_dir is 4 levels up
    build_dir = input_data.get("build_dir")
    if not build_dir:
        bitstream_abs = os.path.abspath(bitstream)
        # Go up: pnrDir -> part_b0_f0 -> fpgaCompDir -> build_dir
        build_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(bitstream_abs))))

    if not os.path.isdir(build_dir):
        ctx.logger.error(f"build_dir not found (inferred from bitstream): {build_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_dir_not_found", "build_dir": build_dir},
            trace_id=origin_tid,
        )
        return

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = input_data.get("log_dir") or f"{bebop_dir}/log/p2e-runworkload-{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # ── Run bebop p2e --runworkload ───────────────────────────────────────
    run_cmd = (
        f"nix develop --ignore-environment --keep HOME --keep ALL_PROXY -c "
        f"cargo run --features p2e "
        f"--config=\"env.OUT_PATH='{build_dir}'\" "
        f"-- p2e "
        f"--runworkload "
        f"--image=\"{image_path}\" "
        f"--bitstream=\"{bitstream}\" "
        f"--build-dir=\"{build_dir}\" "
        f"--log-dir=\"{log_dir}\""
    )
    ctx.logger.info(f"Running bebop p2e runworkload: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e runworkload",
        stderr_prefix="bebop p2e runworkload",
    )

    uart_log = os.path.join(log_dir, "uart.log")
    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "runworkload",
            "image": image_path,
            "bitstream": bitstream,
            "build_dir": build_dir,
            "log_dir": log_dir,
            "uart_log": uart_log,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
