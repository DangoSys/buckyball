"""
bebop p2e runworkload event handler

Loads a kernel image into FPGA and runs the workload via bebop CLI:
  1. Validate image and bitstream paths
  2. Build bebop with p2e feature (VSRC_PATH from previous build, optional)
  3. Run bebop p2e --runworkload --image <image> --bitstream <bitstream>
"""
import os
import sys
from datetime import datetime

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-p2e-runworkload",
    "description": "Run workload on FPGA via bebop p2e CLI",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.runworkload")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    arch_dir = f"{bbdir}/arch"

    image     = input_data.get("image", "")
    bitstream = input_data.get("bitstream", "")
    for label, path in (("image", image), ("bitstream", bitstream)):
        if not path or not os.path.exists(path):
            ctx.logger.error(f"{label} not found: {path}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": f"{label}_not_found", label: path},
                trace_id=origin_tid,
            )
            return

    config_name = input_data.get("config", "sims.p2e.P2EToyConfig")
    vsrc_dir = get_verilator_build_dir(bbdir, config_name, input_data.get("vsrc_dir"))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = input_data.get("output_dir") or os.path.dirname(os.path.abspath(bitstream))
    log_dir    = input_data.get("log_dir")    or f"{arch_dir}/log/p2e-runworkload-{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # ── Build bebop with p2e feature ──────────────────────────────────────
    build_cmd = "cargo build --features p2e"
    if os.path.isdir(vsrc_dir):
        build_cmd += f" --config=\"env.VSRC_PATH='{vsrc_dir}'\""
    ctx.logger.info("Building bebop p2e ...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e build",
        stderr_prefix="bebop p2e build",
    )
    if build_result.returncode != 0:
        await check_result(
            ctx, build_result.returncode, continue_run=False,
            extra_fields={"task": "runworkload", "stage": "build"},
            trace_id=origin_tid,
        )
        return

    # ── Run bebop p2e --runworkload ───────────────────────────────────────
    run_cmd = (
        f"cargo run --features p2e -- p2e "
        f"--runworkload "
        f"--image=\"{image}\" "
        f"--bitstream=\"{bitstream}\" "
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
            "image": image,
            "bitstream": bitstream,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "uart_log": uart_log,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
