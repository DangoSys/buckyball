"""
bebop p2e buildbitstream event handler

Builds the P2E VVAC runtime case via bebop CLI:
  1. Resolve Verilog source directory (VSRC_PATH) from config
  2. Run bebop build p2e with rtl_dir and out_dir
  3. Validate generated bitstream and runtime artifacts
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
    "description": "Build Bebop P2E runtime case",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.buildbitstream")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    config_name = input_data.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        ctx.logger.error("Missing required parameter: config")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_config"},
            trace_id=origin_tid,
        )
        return
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
    build_dir = (
        input_data.get("build_dir")
        or input_data.get("build-dir")
        or input_data.get("output_dir")
        or input_data.get("output-dir")
        or f"{bebop_dir}/build/{config_name}-{timestamp}"
    )
    os.makedirs(build_dir, exist_ok=True)

    build_cmd = (
        f"nix develop --ignore-environment --keep HOME --keep ALL_PROXY -c "
        f"cargo run --features p2e -- build p2e "
        f"--rtl-dir=\"{vsrc_dir}\" "
        f"--out-dir=\"{build_dir}\""
    )
    ctx.logger.info("Building bebop p2e runtime case ...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e build",
        stderr_prefix="bebop p2e build",
    )

    rtcfg_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "rtcfg")
    libvctb_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "lib", "lib_arm", "libvCtb.so")
    bitstream_path = os.path.join(build_dir, "fpgaCompDir", "bitstream.bit")
    if build_result.returncode == 0:
        missing = [
            path
            for path in (rtcfg_path, libvctb_path, bitstream_path)
            if not os.path.exists(path)
        ]
        if missing:
            ctx.logger.error(f"P2E build artifacts missing: {missing}")
            await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={
                    "task": "build",
                    "config": config_name,
                    "vsrc_dir": vsrc_dir,
                    "build_dir": build_dir,
                    "missing": missing,
                    "error": "p2e_artifact_not_found",
                    "timestamp": timestamp,
                },
                trace_id=origin_tid,
            )
            return

    await check_result(
        ctx,
        build_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "build",
            "config": config_name,
            "vsrc_dir": vsrc_dir,
            "build_dir": build_dir,
            "rtcfg": rtcfg_path,
            "libvCtb": libvctb_path,
            "bitstream": bitstream_path,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
