"""
bebop p2e buildbitstream event handler

Builds the FPGA bitstream via bebop CLI:
  1. Resolve verilog source directory (VSRC_PATH) from config
  2. Build bebop with p2e feature (vvacDir generated under build_dir via OUT_PATH)
  3. Run bebop p2e --buildbitstream with build_dir and output_dir
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
    "name": "bebop-p2e-buildbitstream",
    "description": "Build P2E bitstream via bebop CLI",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.buildbitstream")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    arch_dir = f"{bbdir}/arch"

    config_name = input_data.get("config", "sims.p2e.P2EToyConfig")
    vsrc_dir = get_verilator_build_dir(bbdir, config_name, input_data.get("vsrc_dir"))
    if not os.path.isdir(vsrc_dir):
        ctx.logger.error(f"VSRC_PATH does not exist: {vsrc_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "vsrc_not_found", "vsrc_dir": vsrc_dir},
            trace_id=origin_tid,
        )
        return

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    build_dir = input_data.get("build_dir") or f"{bebop_dir}/build/{config_name}-{timestamp}"
    output_dir = input_data.get("output_dir") or f"{arch_dir}/build/p2e-bitstream-{timestamp}"
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Build bebop with p2e feature ──────────────────────────────────────
    build_cmd = (
        f"nix develop --ignore-environment --keep ALL_PROXY -c "
        f"cargo build --features p2e "
        f"--config=\"env.VSRC_PATH='{vsrc_dir}'\" "
        f"--config=\"env.OUT_PATH='{build_dir}'\""
    )
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
            extra_fields={"task": "buildbitstream", "stage": "build"},
            trace_id=origin_tid,
        )
        return

    # ── Run bebop p2e --buildbitstream ────────────────────────────────────
    run_cmd = (
        f"nix develop --ignore-environment --keep HOME --keep ALL_PROXY -c "
        f"cargo run --features p2e "
        f"-- p2e "
        f"--buildbitstream "
        f"--build-dir=\"{build_dir}\" "
        f"--output-dir=\"{output_dir}\""
    )
    ctx.logger.info(f"Running bebop p2e buildbitstream: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e build-bitstream",
        stderr_prefix="bebop p2e build-bitstream",
    )

    bitstream_path = os.path.join(output_dir, "fpgaCompDir", "bitstream.bit")
    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "buildbitstream",
            "config": config_name,
            "vsrc_dir": vsrc_dir,
            "build_dir": build_dir,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "bitstream": bitstream_path,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
