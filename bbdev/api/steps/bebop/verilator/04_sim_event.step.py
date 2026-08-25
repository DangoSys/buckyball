"""
bebop verilator event handler

Run Verilator either through its guest-ELF simulator or the rushB native ABI.
"""
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import require_chip
from utils.path import bebop_target_dir, get_buckyball_path, log_dir, rtl_dir, workload_tests_root
from utils.stream_run import stream_run_logger_async
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id
from build_marker import build_marker_path, read_build_marker


config = {
    "name": "bebop-verilator-sim",
    "description": "Run bebop verilator simulation",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.sim"), queue("bebop.verilator.run.sim")],
    "enqueues": [],
}


def is_native_host_elf(path: str) -> bool:
    try:
        with open(path, "rb") as binary:
            header = binary.read(20)
    except OSError:
        return False
    return header[:4] == b"\x7fELF" and header[18:20] == b"\x3e\x00"


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

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


    diff = bool(input_data.get("diff", False))
    if diff and input_data.get("rushB"):
        ctx.logger.error("--diff and --rushB cannot be used together")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "diff_conflicts_with_rushB"},
            trace_id=origin_tid,
        )
        return

    vsrc_dir = rtl_dir(bbdir, chip, "verilog", input_data.get("vsrc_dir"))
    ctx.logger.info(f"Using verilog source directory: {vsrc_dir}")
    build_dir = bebop_dir
    chip = input_data.get("chip")
    if diff:
        if not isinstance(chip, str) or not chip:
            ctx.logger.error("Missing required parameter: --chip (required with --diff)")
            await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={"error": "missing_chip"},
                trace_id=origin_tid,
            )
            return

    bebop_bin = os.path.join(bebop_target_dir(bbdir, chip), "release", "bebop")
    if not os.path.isfile(bebop_bin):
        ctx.logger.error(f"bebop binary does not exist: {bebop_bin}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "bebop_binary_not_found", "binary": bebop_bin},
            trace_id=origin_tid,
        )
        return

    target_dir = bebop_target_dir(bbdir, chip)
    marker_path = build_marker_path(target_dir)
    try:
        marker = read_build_marker(target_dir)
    except FileNotFoundError:
        ctx.logger.error(f"bebop verilator build marker does not exist: {marker_path}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_marker_not_found", "marker": marker_path},
            trace_id=origin_tid,
        )
        return
    except (OSError, json.JSONDecodeError) as e:
        ctx.logger.error(f"failed to read bebop verilator build marker: {e}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_marker_read_failed", "marker": marker_path, "detail": str(e)},
            trace_id=origin_tid,
        )
        return

    expect_vsrc = os.path.abspath(vsrc_dir)
    expect_bin = os.path.abspath(bebop_bin)
    if (
        marker.get("config") != chip
        or marker.get("vsrc_dir") != expect_vsrc
        or marker.get("binary") != expect_bin
        or bool(marker.get("diff", False)) != diff
    ):
        ctx.logger.error(f"bebop verilator build marker mismatch: {marker}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "build_marker_mismatch",
                "marker": marker,
                "expected": {
                    "vsrc_dir": expect_vsrc,
                    "binary": expect_bin,
                    "diff": diff,
                },
            },
            trace_id=origin_tid,
        )
        return

    binary_name = input_data.get("binary", "")
    workload_root = workload_tests_root(bbdir, chip)
    binary_path = None
    if diff and chip:
        binary_path = search_workload(
            f"{workload_root}/CTest/chips/{chip}", binary_name
        )
    if binary_path is None:
        binary_path = search_workload(workload_root, binary_name)
    if binary_path is None:
        ctx.logger.error(f"binary not found: {binary_name}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "binary_not_found", "binary": binary_name},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"binary_path: {binary_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = log_dir(bbdir, chip, "verilog", timestamp, "verilator", binary_name, input_data.get("vsrc_dir"))
    os.makedirs(run_log, exist_ok=True)

    wave_arg = " --no-wave" if input_data.get("no-wave", False) or input_data.get("no_wave", False) else ""
    trace_args = ""
    for trace_name in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace"):
        if input_data.get(trace_name, False):
            trace_args += f" --{trace_name}"

    if input_data.get("rushB"):
        # rushB owns the Verilator instance inside libbebop_verilator. It must
        # execute as a native program, never as the normal guest ELF input.
        if not is_native_host_elf(binary_path):
            ctx.logger.error(
                f"--rushB requires a native x86_64 rushB runner; got guest ELF: {binary_path}"
            )
            await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={"error": "rushB_requires_native_runner", "binary": binary_path},
                trace_id=origin_tid,
            )
            return
        backend_library = f"{bebop_target_dir(bbdir, chip)}/release/deps/libbebop_verilator.so"
        binary_dir = os.path.dirname(binary_path)
        copy_cmd = shlex.join(["cmake", "-E", "copy_if_different", backend_library, binary_dir])
        inner_cmd = f"cd {shlex.quote(binary_dir)} && {copy_cmd} && exec {shlex.quote(binary_path)}"
        run_cmd = f"nix develop -c sh -c {shlex.quote(inner_cmd)}"
        ctx.logger.info(f"Running rushB Verilator: {run_cmd}")
        stdout_prefix = "rushB verilator"
        stderr_prefix = "rushB verilator"
    else:
        run_cmd = (
            f"{shlex.quote(bebop_bin)} run verilator "
            f"--elf={shlex.quote(binary_path)} "
            f"--log-dir={shlex.quote(run_log)}"
            f"{' --diff' if diff else ''}"
            f"{wave_arg}"
            f"{trace_args}"
        )
        if diff:
            preload = os.pathsep.join(
                path
                for path in (
                    f"{bbdir}/result/lib/libdramsim3.so",
                    os.environ.get("LD_PRELOAD", ""),
                )
                if path
            )
            run_inner = (
                f"cd {shlex.quote(build_dir)} && "
                f"export LD_PRELOAD={shlex.quote(preload)} && exec {run_cmd}"
            )
            run_cmd = f"nix develop -c sh -c {shlex.quote(run_inner)}"
        ctx.logger.info(f"Running bebop verilator (diff={diff}): {run_cmd}")
        stdout_prefix = "bebop verilator difftest" if diff else "bebop verilator"
        stderr_prefix = stdout_prefix
    run_result = await stream_run_logger_async(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix=stdout_prefix,
        stderr_prefix=stderr_prefix,
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilator",
            "backend": "rushB" if input_data.get("rushB") else "guest-elf",
            "diff": diff,
            "chip": chip,
            "binary": binary_path,
            "log_dir": run_log,
            "bank_diff": os.path.join(run_log, "bank_diff.ndjson") if diff else None,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
