"""
bebop bemu event handler

Runs bebop bemu (Spike-based) emulator:
  1. Resolve binary (ELF) path
  2. Build bebop with bemu feature (cargo build)
  3. Run bebop bemu with the resolved ELF
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

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id

PERFETTO_TARGETS = {
    "buddy-buckyball-lenet-run": "buddy-buckyball-lenet-perfetto",
}

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


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    arch_dir = f"{bbdir}/arch"

    binary_name = input_data.get("binary", "")
    binary_path = search_workload(f"{bbdir}/bb-tests/output/workloads/src", binary_name)
    if binary_path is None:
        ctx.logger.error(f"binary not found: {binary_name}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "binary_not_found", "binary": binary_name},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"binary_path: {binary_path}")
    binary_dir = os.path.dirname(binary_path)
    perfetto_target = PERFETTO_TARGETS.get(binary_name)
    if perfetto_target:
        clean_model_trace(binary_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = input_data.get("log_dir") or f"{arch_dir}/log/{timestamp}-{binary_name}-bemu"
    os.makedirs(log_dir, exist_ok=True)

    # ── Run bebop bemu ────────────────────────────────────────────────────
    pk_flag = " --pk" if input_data.get("pk") else ""
    run_cmd = (
        f"cargo run --manifest-path \"{bebop_dir}/Cargo.toml\" --features bemu -- bemu "
        f"--elf=\"{binary_path}\" "
        f"--log-dir=\"{log_dir}\""
        f"{pk_flag}"
    )
    ctx.logger.info(f"Running bebop bemu: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=binary_dir,
        stdout_prefix="bebop bemu",
        stderr_prefix="bebop bemu",
    )
    if run_result.returncode != 0:
        await check_result(
            ctx,
            run_result.returncode,
            continue_run=False,
            extra_fields={
                "task": "bemu",
                "binary": binary_path,
                "log_dir": log_dir,
                "timestamp": timestamp,
            },
            trace_id=origin_tid,
        )
        return

    perfetto_path = None
    if perfetto_target:
        perfetto_cmd = (
            f"cmake --build {shlex.quote(f'{bbdir}/bb-tests/build')} "
            f"--target {shlex.quote(perfetto_target)}"
        )
        ctx.logger.info(f"Generating Perfetto trace: {perfetto_cmd}")
        perfetto_result = stream_run_logger(
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
                    "log_dir": log_dir,
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
            "log_dir": log_dir,
            "timestamp": timestamp,
            "perfetto_target": perfetto_target,
            "perfetto": perfetto_path,
        },
        trace_id=origin_tid,
    )
