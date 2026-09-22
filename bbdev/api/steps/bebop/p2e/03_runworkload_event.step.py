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
import shlex
import sys
from datetime import datetime

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import require_chip
from utils.path import bebop_cargo_env, get_buckyball_path, log_dir
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from resolve_image import resolve_image

config = {
    "name": "bebop-p2e-runworkload",
    "description": "Run workload on FPGA via bebop p2e CLI",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.runworkload")],
    "enqueues": [],
}


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
    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    image_name = input_data.get("image", "")
    bitstream = input_data.get("bitstream", "")
    multi_fpga = bool(input_data.get("multi-fpga", False))
    fpga_location = input_data.get("fpga-location") or input_data.get("fpga_location") or "0.A"
    if not isinstance(fpga_location, str) or not re.fullmatch(r"[0-3]\.[A-E]", fpga_location):
        ctx.logger.error(f"invalid FPGA location: {fpga_location}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_fpga_location", "fpga_location": fpga_location},
            trace_id=origin_tid,
        )
        return
    wave = bool(input_data.get("wave", False))
    diff = bool(input_data.get("diff", False))
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

    try:
        image_path = resolve_image(bbdir, image_name, chip)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "image_not_found", "image": image_name},
            trace_id=origin_tid,
        )
        return

    image_base = os.path.splitext(image_path)[0]
    elf_path = f"{image_base}.elf" if os.path.isfile(f"{image_base}.elf") else image_base
    if diff and not os.path.isfile(elf_path):
        ctx.logger.error(f"workload ELF for P2E DiffTest not found: {elf_path}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "workload_elf_not_found", "elf": elf_path},
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

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = log_dir(bbdir, chip, "p2e", timestamp, "p2e", image_name, input_data.get("vsrc_dir"))
    os.makedirs(run_log, exist_ok=True)

    runtime_lib_dir = os.path.join(build_dir, "vvacDir", "runtimeDir", "lib", "lib_arm")
    rtcfg_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "rtcfg")
    libvctb_path = os.path.join(runtime_lib_dir, "libvCtb.so")
    bebop_p2e_path = os.path.join(build_dir, "bebop-p2e")
    runtime_artifacts = [rtcfg_path, libvctb_path, bebop_p2e_path]
    if diff:
        runtime_artifacts.append(os.path.join(build_dir, "libriscv.so"))
    if not all(os.path.isfile(path) for path in runtime_artifacts):
        missing = [path for path in runtime_artifacts if not os.path.isfile(path)]
        if missing:
            ctx.logger.error(f"P2E runtime artifacts missing: {missing}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "runtime",
                "build_dir": build_dir,
                "missing": missing,
            },
            trace_id=origin_tid,
        )
        return

    run_cmd = (
        f"\"{bebop_p2e_path}\" run p2e "
        f"--image=\"{image_path}\" "
        f"--bitstream=\"{bitstream}\" "
        f"--log-dir=\"{run_log}\" "
        f"--fpga-location=\"{fpga_location}\""
    )
    if multi_fpga:
        run_cmd += " --multi-fpga"
    if wave:
        run_cmd += " --wave"
    if wave_start is not None:
        run_cmd += f" --wave-start=\"{wave_start}\""
    if diff:
        run_cmd += f" --diff --image-elf={shlex.quote(elf_path)}"
    for trace_name in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace"):
        if input_data.get(trace_name, False):
            run_cmd += f" --{trace_name}"
    ctx.logger.info(f"Running bebop p2e runworkload: {run_cmd}")
    run_env = {**os.environ.copy(), **bebop_cargo_env(bbdir, chip)}
    run_result = await stream_run_logger_async(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e runworkload",
        stderr_prefix="bebop p2e runworkload",
        env=run_env,
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "runworkload",
            "image": image_path,
            "bitstream": bitstream,
            "build_dir": build_dir,
            "log_dir": run_log,
            "fpga_location": fpga_location,
            "bdb_trace": os.path.join(run_log, "bdb.ndjson"),
            "uart_log": os.path.join(run_log, "uart.log"),
            "bank_diff": os.path.join(run_log, "diff.ndjson") if diff else None,
            "diff": diff,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
