"""
bebop bemu event handler

Run a guest ELF through BEMU.
"""

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
from utils.path import (
    bebop_cargo_env,
    get_buckyball_path,
    log_dir,
    workloads_output_root,
)
from utils.stream_run import stream_run_logger_async
from utils.search_workload import search_workload
from steps.bebop.bemu.scripts.model_sim import model_run_commands
from utils.event_common import check_result, get_origin_trace_id
from utils.process_registry import cancellation_requested
from bemu_common import bemu_chip_binary, bemu_manifest, bemu_tile_index
from steps.bebop.performance_report import performance_report
from utils.reports import source_context

config = {
    "name": "bebop-bemu-sim",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.sim")],
    "enqueues": [],
}


def clean_model_trace(binary_dir: str) -> None:
    trace_dir = Path(binary_dir) / "trace"
    for subdir in ("cycle", "tensor"):
        target_dir = trace_dir / subdir
        if not target_dir.exists():
            continue
        if not target_dir.is_dir():
            raise NotADirectoryError(f"trace path is not a directory: {target_dir}")
        for path in target_dir.glob("trace-*.txt"):
            if not path.is_file():
                raise FileNotFoundError(f"trace path is not a file: {path}")
            path.unlink()
        summary = target_dir / "summary.txt"
        if summary.exists():
            if not summary.is_file():
                raise FileNotFoundError(f"trace summary path is not a file: {summary}")
            summary.unlink()
    perfetto = trace_dir / "perfetto.json"
    if perfetto.exists():
        if not perfetto.is_file():
            raise FileNotFoundError(f"perfetto path is not a file: {perfetto}")
        perfetto.unlink()


def resolve_bemu_binary(bbdir: str, chip: str, binary_name: str) -> str | None:
    if Path(binary_name).name != binary_name:
        return None
    workload_root = Path(workloads_output_root(bbdir)) / chip / "workloads"
    workload = search_workload(str(workload_root), binary_name)
    if workload is not None:
        return workload
    kernel = Path(bbdir) / "bb-tests" / "output" / "kernel" / chip / binary_name
    return str(kernel) if kernel.is_file() else None


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip must be specified")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    try:
        bemu_cargo_manifest = bemu_manifest(chip, bbdir)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return
    if input_data.get("model"):
        report_context = source_context(bbdir, chip)
        params = {key: value for key, value in input_data.items() if key != "_trace_id"}
        commands, run_log = model_run_commands(bbdir, params)
        run_log.mkdir(parents=True, exist_ok=True)
        ctx.logger.info(f"Model simulation logs: {run_log}")
        filenames = (
            ("run.log",)
            if input_data.get("reuse-simulator", False)
            else ("build.log", "run.log")
        )
        for command, filename in zip(commands, filenames, strict=True):
            command_line = shlex.join(["nix", "develop", "-c", *command])
            command_line = f"set -o pipefail; {command_line} 2>&1 | tee {shlex.quote(str(run_log / filename))}"
            result = await stream_run_logger_async(
                cmd=command_line,
                executable="/bin/bash",
                logger=ctx.logger,
                cwd=bbdir,
                task_scope=origin_tid,
                stdout_prefix="model simulation",
                stderr_prefix="model simulation",
            )
            if cancellation_requested(origin_tid):
                return
            if result.returncode:
                break
        try:
            report_path = performance_report(
                bbdir,
                report_context,
                origin_tid,
                "bemu",
                run_log,
                (run_log / "run.log").read_text(),
                result.returncode,
            )
        except (OSError, ValueError, KeyError) as error:
            await check_result(
                ctx,
                1,
                extra_fields={
                    "task": "report",
                    "error": str(error),
                    "log_dir": str(run_log),
                },
                trace_id=origin_tid,
            )
            return
        await check_result(
            ctx,
            result.returncode,
            continue_run=False,
            extra_fields={
                "chip": chip,
                "model": params["model"],
                "report_path": report_path,
                "log_dir": str(run_log),
            },
            trace_id=origin_tid,
        )
        return
    binary_name = input_data.get("binary", "")
    try:
        binary_path = resolve_bemu_binary(bbdir, chip, binary_name)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "binary_resolution_error", "binary": binary_name},
            trace_id=origin_tid,
        )
        return
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
    perfetto_target = None
    if input_data.get("tool-profile"):
        if not binary_name.endswith("-run"):
            raise ValueError(f"tool-profile binary must end with '-run': {binary_name}")
        perfetto_target = f"{binary_name.removesuffix('-run')}-perfetto"
    if perfetto_target:
        clean_model_trace(binary_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = log_dir(bbdir, chip, "verilog", timestamp, "bemu", binary_name)
    os.makedirs(run_log, exist_ok=True)
    core_index = input_data.get("core_index")
    system = bool(input_data.get("system", False))
    if system and core_index is not None:
        raise ValueError(
            "system boot runs all chip harts; core_index cannot be supplied"
        )
    # The chip's tile runner by default; --core-index selects the single-core bebop-bemu.
    if core_index is None:
        entry = [bemu_chip_binary(chip), "--"]
        if not system:
            entry.extend(["--tile-index", str(bemu_tile_index(chip, bbdir))])
    else:
        entry = ["bebop-bemu", "--"]
    cargo_args = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(bemu_cargo_manifest),
        "--bin",
        *entry,
        "--elf",
        binary_path,
        "--log-dir",
        run_log,
    ]
    cargo_args[1:1] = [
        "--config",
        'build.rustflags=["-C", "target-cpu=native"]',
        "--config",
        'profile.release.lto="thin"',
        "--config",
        "profile.release.codegen-units=1",
    ]
    if system:
        cargo_args.append("--system")
        for option in ("dtb", "initrd"):
            if input_data.get(option):
                cargo_args.extend([f"--{option}", input_data[option]])
        if input_data.get("memory-mib") is not None:
            cargo_args.extend(["--memory-mib", str(input_data["memory-mib"])])
    if core_index is not None:
        cargo_args.extend(["--core-index", str(core_index)])
    if input_data.get("disasm"):
        cargo_args.append("--disasm")
    if input_data.get("tool-profile"):
        cargo_args.append("--tool-profile")
    for trace_name in ("itrace", "mtrace"):
        if input_data.get(trace_name, False):
            cargo_args.append(f"--{trace_name}")
    arguments = input_data.get("arguments", [])
    if arguments:
        if not isinstance(arguments, list) or not all(
            isinstance(arg, str) for arg in arguments
        ):
            raise ValueError("arguments must be a list of strings")
        cargo_args.extend(["--", *arguments])
    inner_cmd = f"cd {shlex.quote(binary_dir)} && {shlex.join(cargo_args)}"
    run_cmd = f"nix develop -c sh -c {shlex.quote(inner_cmd)}"
    run_cmd = f"set -o pipefail; {run_cmd} 2>&1 | tee {shlex.quote(str(Path(run_log) / 'run.log'))}"
    ctx.logger.info(f"Running bebop bemu: {run_cmd}")
    report_context = source_context(bbdir, chip)
    run_result = await stream_run_logger_async(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        executable="/bin/bash",
        stdout_prefix="bebop bemu",
        stderr_prefix="bebop bemu",
        task_scope=origin_tid,
        env={**os.environ.copy(), **bebop_cargo_env(bbdir, chip)},
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
                "chip": chip,
                "log_dir": run_log,
                "timestamp": timestamp,
            },
            trace_id=origin_tid,
        )
        return
    perfetto_path = None
    if perfetto_target:
        perfetto_cmd = f"cmake --build {shlex.quote(f'{bbdir}/bb-tests/build')} --target {shlex.quote(perfetto_target)}"
        ctx.logger.info(f"Generating Perfetto trace: {perfetto_cmd}")
        perfetto_result = await stream_run_logger_async(
            cmd=perfetto_cmd,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="perfetto",
            stderr_prefix="perfetto",
        )
        perfetto_path = f"{binary_dir}/trace/perfetto.json"
        if perfetto_result.returncode != 0:
            await check_result(
                ctx,
                perfetto_result.returncode,
                continue_run=False,
                extra_fields={
                    "task": "perfetto",
                    "binary": binary_path,
                    "log_dir": run_log,
                    "timestamp": timestamp,
                    "perfetto_target": perfetto_target,
                    "perfetto": perfetto_path,
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
            "chip": chip,
            "log_dir": run_log,
            "timestamp": timestamp,
            "perfetto_target": perfetto_target,
            "perfetto": perfetto_path,
        },
        trace_id=origin_tid,
    )
