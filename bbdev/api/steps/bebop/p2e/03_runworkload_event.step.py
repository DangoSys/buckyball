"""
bebop p2e runworkload event handler

Loads a kernel image into FPGA and runs the workload via bebop CLI:
  1. Resolve image name to .hex file under bb-tests/output/
  2. Validate bitstream .bit file path
  3. Run bebop run p2e --image <image-path> --bitstream <bitstream> [--multi-fpga] [--wave] [--wave-start <cycle>]
"""
import glob
import os
import re
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


def resolve_image(bbdir: str, image_name: str) -> str:
    """Search bb-tests/output/ recursively for <image_name>.hex and return its absolute path."""
    output_root = f"{bbdir}/bb-tests/output"
    matches = glob.glob(f"{output_root}/**/{image_name}.hex", recursive=True)
    if not matches:
        return ""
    return matches[0]


def resolve_runtime_config(bitstream: str, requested_config: object) -> str:
    if isinstance(requested_config, str) and requested_config:
        return requested_config

    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(bitstream)))
    case_name = os.path.basename(build_dir)
    return re.sub(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$", "", case_name)


def case_uses_multi_fpga(build_dir: str) -> bool:
    fpga_comp_dir = os.path.join(build_dir, "fpgaCompDir")
    part_dirs = glob.glob(os.path.join(fpga_comp_dir, "part_b*_f*"))
    return len([path for path in part_dirs if os.path.isdir(path)]) > 1


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    image_name = input_data.get("image", "")
    bitstream = input_data.get("bitstream", "")
    multi_fpga = bool(input_data.get("multi-fpga", False))
    wave = bool(input_data.get("wave", False))
    if "wave_start" in input_data:
        ctx.logger.error("invalid parameter: --wave_start (use --wave-start)")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_parameter", "parameter": "wave_start"},
            trace_id=origin_tid,
        )
        return
    wave_start_raw = input_data.get("wave-start")
    wave_start = None
    if wave_start_raw is not None:
        try:
            wave_start = int(wave_start_raw)
        except (TypeError, ValueError):
            ctx.logger.error(f"invalid wave_start: {wave_start_raw}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "invalid_wave_start", "wave_start": wave_start_raw},
                trace_id=origin_tid,
            )
            return
        if wave_start < 0:
            ctx.logger.error(f"wave_start must be >= 0: {wave_start}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "invalid_wave_start", "wave_start": wave_start},
                trace_id=origin_tid,
            )
            return
        wave = True

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

    bitstream = os.path.abspath(bitstream)
    build_dir = os.path.dirname(os.path.dirname(bitstream))
    if not os.path.isdir(build_dir):
        ctx.logger.error(f"P2E build case not found for bitstream: {build_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_dir_not_found", "build_dir": build_dir},
            trace_id=origin_tid,
        )
        return

    if not multi_fpga and case_uses_multi_fpga(build_dir):
        multi_fpga = True
        ctx.logger.info(f"Detected multi-FPGA P2E case: {build_dir}")

    config_name = resolve_runtime_config(bitstream, input_data.get("config"))
    vsrc_dir = get_verilator_build_dir(bbdir, config_name, input_data.get("vsrc_dir"))
    if not os.path.isdir(vsrc_dir):
        ctx.logger.error(f"VSRC_PATH does not exist for P2E runtime: {vsrc_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "vsrc_not_found",
                "config": config_name,
                "vsrc_dir": vsrc_dir,
            },
            trace_id=origin_tid,
        )
        return

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = f"{bbdir}/log/{timestamp}-p2e-{image_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Rebuild the VVAC host runtime in the bitstream case.  The bitstream is
    # deliberately left in place, so runtime/DPIC changes never trigger FPGA synthesis.
    runtime_cmd = (
        f"env BEBOP_P2E_RUNTIME_ONLY=1 BEBOP_P2E_REBUILD_RUNTIME=1 "
        f"cargo run --features p2e -- build p2e "
        f"--rtl-dir=\"{vsrc_dir}\" "
        f"--out-dir=\"{build_dir}\""
    )
    ctx.logger.info("Preparing bebop p2e runtime for the selected bitstream ...")
    runtime_result = stream_run_logger(
        cmd=runtime_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e runtime",
        stderr_prefix="bebop p2e runtime",
    )
    rtcfg_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "rtcfg")
    libvctb_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "lib", "lib_arm", "libvCtb.so")
    if runtime_result.returncode != 0 or not all(os.path.isfile(path) for path in (rtcfg_path, libvctb_path)):
        missing = [path for path in (rtcfg_path, libvctb_path) if not os.path.isfile(path)]
        if missing:
            ctx.logger.error(f"P2E runtime artifacts missing: {missing}")
        await check_result(
            ctx,
            runtime_result.returncode or 1,
            continue_run=False,
            extra_fields={
                "task": "runtime",
                "config": config_name,
                "vsrc_dir": vsrc_dir,
                "build_dir": build_dir,
                "missing": missing,
            },
            trace_id=origin_tid,
        )
        return

    # ── Run bebop run p2e ─────────────────────────────────────────────────
    run_cmd = (
        f"cargo run --features p2e "
        f"--config=\"env.OUT_PATH='{build_dir}'\" "
        f"-- run p2e "
        f"--image=\"{image_path}\" "
        f"--bitstream=\"{bitstream}\" "
        f"--log-dir=\"{log_dir}\""
    )
    if multi_fpga:
        run_cmd += " --multi-fpga"
    if wave:
        run_cmd += " --wave"
    if wave_start is not None:
        run_cmd += f" --wave-start=\"{wave_start}\""
    for trace_name in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace"):
        if input_data.get(trace_name, False):
            run_cmd += f" --{trace_name}"
    ctx.logger.info(f"Running bebop p2e runworkload: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e runworkload",
        stderr_prefix="bebop p2e runworkload",
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "runworkload",
            "image": image_path,
            "bitstream": bitstream,
            "config": config_name,
            "build_dir": build_dir,
            "log_dir": log_dir,
            "bdb_trace": os.path.join(log_dir, "bdb.ndjson"),
            "uart_log": os.path.join(log_dir, "uart.log"),
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
