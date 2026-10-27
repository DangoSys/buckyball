"""
bebop bemu event handler

Run a guest ELF through BEMU.
"""

import os
import shlex
import shutil
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
from utils.path import (
    bebop_target_dir,
    bebop_bin_path,
    get_buckyball_path,
    log_dir,
    design_output_root,
)
from utils.stream_run import stream_run_logger_async
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id
from utils.process_registry import cancellation_requested
from bemu_common import bemu_binary, bemu_manifest
from steps.bebop.performance_report import performance_report
from utils.reports import source_context

config = {
    "name": "bebop-bemu-sim",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.sim")],
    "enqueues": [],
}


def resolve_bemu_binary(
    bbdir: str, design: str, binary_name: str, selection: str
) -> str | None:
    workload_root = (
        Path(design_output_root(bbdir, design, selection=selection)) / "workloads"
    )
    return search_workload(str(workload_root), binary_name)


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    design = input_data["design"]
    bemu_cargo_manifest = bemu_manifest(design, bbdir, selection=input_data["target"])
    binary_name = input_data["binary"]
    binary_path = resolve_bemu_binary(bbdir, design, binary_name, input_data["target"])
    if binary_path is None:
        ctx.logger.error(f"binary not found: {binary_name}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "binary_not_found", "binary": binary_name},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"binary_path: {binary_path}")
    binary_dir = os.path.dirname(binary_path)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = log_dir(
        bbdir,
        design,
        "verilog",
        timestamp,
        "bemu",
        binary_name,
        selection=input_data["target"],
    )
    os.makedirs(run_log, exist_ok=True)
    target = Path(
        bebop_target_dir(bbdir, design, "bemu", selection=input_data["target"])
    )
    installed = Path(
        bebop_bin_path(bbdir, design, "bemu", selection=input_data["target"])
    )
    build_args = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(bemu_cargo_manifest),
        "--target-dir",
        str(target),
        "--bin",
        bemu_binary(),
        "--features",
        "bemu",
        "--config",
        'build.rustflags=["-C", "target-cpu=native"]',
        "--config",
        'profile.release.lto="thin"',
        "--config",
        "profile.release.codegen-units=1",
    ]
    build_result = await stream_run_logger_async(
        cmd=shlex.join(["nix", "develop", "-c", *build_args]),
        executable="/bin/bash",
        logger=ctx.logger,
        cwd=bbdir,
        task_scope=origin_tid,
        stdout_prefix="BEMU build",
        stderr_prefix="BEMU build",
    )
    if cancellation_requested(origin_tid):
        return
    if build_result.returncode:
        await check_result(
            ctx,
            build_result.returncode,
            continue_run=False,
            extra_fields={"task": "build", "design": design},
            trace_id=origin_tid,
        )
        return
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release/bebop-bemu", installed)
    run_args = [str(installed), "--elf", binary_path, "--log-dir", run_log]
    # `bbdev kernel --build` lays the kernel and the model into <name>-ddr.bin
    # next to the firmware ELF.
    ddr_image = Path(binary_path).with_suffix("").as_posix() + "-ddr.bin"
    if os.path.isfile(ddr_image):
        run_args.extend(["--ddr-image", ddr_image])
    for option in ("dtb", "initrd"):
        if input_data.get(option):
            run_args.extend([f"--{option}", input_data[option]])
    if input_data.get("memory-mib") is not None:
        run_args.extend(["--memory-mib", str(input_data["memory-mib"])])
    if input_data.get("disasm"):
        run_args.append("--disasm")
    if input_data.get("tool-profile"):
        run_args.append("--tool-profile")
    for trace_name in ("itrace", "mtrace", "ctrace"):
        if input_data.get(trace_name, False):
            run_args.append(f"--{trace_name}")
    inner_cmd = f"cd {shlex.quote(binary_dir)} && {shlex.join(run_args)}"
    run_cmd = f"nix develop -c sh -c {shlex.quote(inner_cmd)}"
    run_cmd = f"set -o pipefail; {run_cmd} 2>&1 | tee {shlex.quote(str(Path(run_log) / 'run.log'))}"
    ctx.logger.info(f"Running bebop bemu: {run_cmd}")
    report_context = source_context(bbdir, design, input_data["target"])
    run_result = await stream_run_logger_async(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        executable="/bin/bash",
        stdout_prefix="bebop bemu",
        stderr_prefix="bebop bemu",
        task_scope=origin_tid,
        env=os.environ.copy(),
    )
    if cancellation_requested(origin_tid):
        return
    try:
        report_path = performance_report(
            bbdir,
            report_context,
            origin_tid,
            "bemu",
            run_log,
            (Path(run_log) / "run.log").read_text(),
            run_result.returncode,
        )
    except (OSError, ValueError, KeyError) as error:
        await check_result(
            ctx,
            1,
            extra_fields={"task": "report", "error": str(error), "log_dir": run_log},
            trace_id=origin_tid,
        )
        return
    if run_result.returncode != 0:
        await check_result(
            ctx,
            run_result.returncode,
            continue_run=False,
            extra_fields={
                "task": "bemu",
                "report_path": report_path,
                "binary": binary_path,
                "design": design,
                "log_dir": run_log,
                "timestamp": timestamp,
            },
            trace_id=origin_tid,
        )
        return
    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "task": "bemu",
            "report_path": report_path,
            "binary": binary_path,
            "design": design,
            "log_dir": run_log,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
