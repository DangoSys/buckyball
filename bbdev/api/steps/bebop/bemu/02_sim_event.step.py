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
from bemu_common import bemu_manifest, bemu_tile_index, chip_emu_manifest

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
    workload = search_workload(workloads_output_root(bbdir), binary_name)
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
        params = {key: value for key, value in input_data.items() if key != "_trace_id"}
        commands, run_log = model_run_commands(bbdir, params)
        run_log.mkdir(parents=True, exist_ok=True)
        ctx.logger.info(f"Model simulation logs: {run_log}")
        for command, filename in zip(commands, ("build.log", "run.log"), strict=True):
            command_line = shlex.join(["nix", "develop", "-c", *command])
            command_line = f"set -o pipefail; {command_line} 2>&1 | tee {shlex.quote(str(run_log / filename))}"
            result = await stream_run_logger_async(
                cmd=command_line, executable="/bin/bash",
                logger=ctx.logger, cwd=bbdir, task_scope=origin_tid,
                stdout_prefix="model simulation", stderr_prefix="model simulation")
            if cancellation_requested(origin_tid):
                return
            if result.returncode:
                break
        await check_result(ctx, result.returncode, continue_run=False,
            extra_fields={"chip": chip, "model": params["model"], "log_dir": str(run_log)},
            trace_id=origin_tid)
        return
    binary_name = input_data.get("binary", "")
    binary_path = resolve_bemu_binary(bbdir, chip, binary_name)
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
    tile_index = bemu_tile_index(chip, bbdir)
    chip_emu = chip_emu_manifest(chip, bbdir)
    if (tile_index is not None) != bool(chip_emu):
        ctx.logger.error(
            f"chip {chip}: bundle bemu.chipMain and emu/Cargo.toml must both exist or both be absent"
        )
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "chip_emu_entry_mismatch", "chip": chip},
            trace_id=origin_tid,
        )
        return
    if chip_emu and core_index is None:
        cargo_args = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(chip_emu),
            "--",
            "--tile-index",
            str(tile_index),
            "--elf",
            binary_path,
            "--log-dir",
            run_log,
        ]
    else:
        cargo_args = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(bemu_cargo_manifest),
            "--bin",
            "bebop-bemu",
            "--",
            "--elf",
            binary_path,
            "--log-dir",
            run_log,
        ]
    if core_index is not None:
        cargo_args.extend(["--core-index", str(core_index)])
    if input_data.get("pk"):
        cargo_args.append("--pk")
    if input_data.get("host-io"):
        cargo_args.append("--host-io")
    if input_data.get("disasm"):
        cargo_args.append("--disasm")
    if input_data.get("tool-profile"):
        cargo_args.append("--tool-profile")
    for trace_name in ("itrace", "mtrace"):
        if input_data.get(trace_name, False):
            cargo_args.append(f"--{trace_name}")
    arguments = input_data.get("arguments", [])
    if arguments:
        if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
            raise ValueError("arguments must be a list of strings")
        cargo_args.extend(["--", *arguments])
    inner_cmd = f"cd {shlex.quote(binary_dir)} && {shlex.join(cargo_args)}"
    run_cmd = f"nix develop -c sh -c {shlex.quote(inner_cmd)}"
    run_cmd = f"set -o pipefail; {run_cmd} 2>&1 | tee {shlex.quote(str(Path(run_log) / 'run.log'))}"
    ctx.logger.info(f"Running bebop bemu: {run_cmd}")
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
    if run_result.returncode != 0:
        await check_result(
            ctx,
            run_result.returncode,
            continue_run=False,
            extra_fields={
                "task": "bemu",
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
            "binary": binary_path,
            "chip": chip,
            "log_dir": run_log,
            "timestamp": timestamp,
            "perfetto_target": perfetto_target,
            "perfetto": perfetto_path,
        },
        trace_id=origin_tid,
    )
