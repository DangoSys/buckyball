"""Run Bank DiffTest on Verilator RTL using BEMU as the reference model."""

import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.search_workload import search_workload, search_workload_all
from utils.stream_run import stream_run_logger


config = {
    "name": "difftest-run",
    "description": "Run Bank DiffTest on Verilator RTL with BEMU as the reference model",
    "flows": ["difftest"],
    "triggers": [queue("difftest.run")],
    "enqueues": [],
}


def resolve_workload(bbdir: str, chip: str, binary: str) -> str | None:
    candidate = Path(binary)
    if not candidate.is_absolute():
        candidate = Path(bbdir) / candidate
    if candidate.is_file():
        return str(candidate.resolve())

    search_root = f"{bbdir}/bb-tests/output/workloads/src"
    chip_match = search_workload(f"{search_root}/CTest/chips/{chip}", binary)
    if chip_match is not None:
        return chip_match

    matches = search_workload_all(search_root, binary)
    chip_marker = f"{os.path.sep}CTest{os.path.sep}chips{os.path.sep}"
    non_chip_matches = [path for path in matches if chip_marker not in path]
    return non_chip_matches[0] if len(non_chip_matches) == 1 else None


async def fail(ctx, trace_id: str, error: str, **fields) -> None:
    ctx.logger.error(fields.pop("message", error))
    await check_result(
        ctx,
        1,
        continue_run=False,
        extra_fields={"error": error, **fields},
        trace_id=trace_id,
    )


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    chip = input_data["chip"]
    arch_config = input_data["config"]
    binary_name = input_data["binary"]

    chip_emu_dir = Path(bbdir) / "examples" / "chips" / chip / "emu"
    manifest = chip_emu_dir / "Cargo.toml"
    if not manifest.is_file():
        await fail(ctx, origin_tid, "invalid_chip", chip=chip, manifest=str(manifest))
        return

    vsrc_dir = os.path.abspath(input_data["vsrc_dir"])
    if not os.path.isdir(vsrc_dir):
        await fail(
            ctx,
            origin_tid,
            "vsrc_not_found",
            config=arch_config,
            vsrc_dir=vsrc_dir,
            message=f"RTL directory does not exist: {vsrc_dir}; generate Verilog first",
        )
        return

    binary_path = resolve_workload(bbdir, chip, binary_name)
    if binary_path is None:
        await fail(ctx, origin_tid, "binary_not_found", chip=chip, binary=binary_name)
        return

    try:
        jobs = int(input_data.get("jobs", 16))
        if jobs <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await fail(ctx, origin_tid, "invalid_jobs", jobs=input_data.get("jobs"))
        return

    # sourceme.sh points RISCV at the repository's `result` environment.  Keep
    # it current when a newly required native dependency is not present.
    dramsim_header = Path(bbdir) / "result" / "include" / "dramsim3.h"
    if not dramsim_header.is_file():
        ctx.logger.info(
            f"Nix environment is missing {dramsim_header}; rebuilding result"
        )
        env_result = stream_run_logger(
            cmd="nix build",
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="difftest environment",
            stderr_prefix="difftest environment",
        )
        if env_result.returncode != 0:
            await check_result(
                ctx,
                env_result.returncode,
                continue_run=False,
                extra_fields={
                    "task": "environment",
                    "backend": "verilator",
                    "chip": chip,
                    "config": arch_config,
                    "dependency": str(dramsim_header),
                },
                trace_id=origin_tid,
            )
            return

    bebop_bin = chip_emu_dir / "target" / "release" / "bebop"
    cargo_build_cmd = shlex.join(
        [
            "cargo",
            "build",
            "--release",
            "--bin",
            "bebop",
            "--features",
            "verilator,bemu,difftest",
            "--jobs",
            str(jobs),
        ]
    )
    build_inner = f"cd {shlex.quote(str(chip_emu_dir))} && exec {cargo_build_cmd}"
    build_cmd = f"nix develop -c sh -c {shlex.quote(build_inner)}"
    ctx.logger.info(f"Building Chip DiffTest executable: {build_cmd}")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="difftest build",
        stderr_prefix="difftest build",
        env={**os.environ, "VSRC_PATH": vsrc_dir},
    )
    if build_result.returncode != 0:
        await check_result(
            ctx,
            build_result.returncode,
            continue_run=False,
            extra_fields={
                "task": "build",
                "backend": "verilator",
                "chip": chip,
                "config": arch_config,
                "binary": str(bebop_bin),
                "vsrc_dir": vsrc_dir,
            },
            trace_id=origin_tid,
        )
        return

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    requested_log_dir = input_data.get("log_dir")
    if requested_log_dir:
        log_dir = os.path.abspath(os.path.join(bbdir, requested_log_dir))
    else:
        log_dir = f"{bbdir}/log/{timestamp}-difftest-{binary_name}"
    os.makedirs(log_dir, exist_ok=True)

    run_args = [
        str(bebop_bin),
        "run",
        "verilator",
        f"--elf={binary_path}",
        f"--log-dir={log_dir}",
        "--diff",
    ]
    if input_data.get("no_wave"):
        run_args.append("--no-wave")
    for trace_name in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace"):
        if input_data.get(trace_name):
            run_args.append(f"--{trace_name}")

    preload = os.pathsep.join(
        path
        for path in (
            f"{bbdir}/result/lib/libdramsim3.so",
            os.environ.get("LD_PRELOAD", ""),
        )
        if path
    )
    run_inner = (
        f"cd {shlex.quote(str(chip_emu_dir))} && "
        f"export LD_PRELOAD={shlex.quote(preload)} && exec {shlex.join(run_args)}"
    )
    run_cmd = f"nix develop -c sh -c {shlex.quote(run_inner)}"
    ctx.logger.info(f"Running Bank DiffTest: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="difftest",
        stderr_prefix="difftest",
    )
    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "difftest",
            "backend": "verilator",
            "chip": chip,
            "config": arch_config,
            "binary": binary_path,
            "vsrc_dir": vsrc_dir,
            "log_dir": log_dir,
            "bank_diff": os.path.join(log_dir, "bank_diff.ndjson"),
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
