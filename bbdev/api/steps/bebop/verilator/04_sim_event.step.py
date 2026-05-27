"""
bebop verilator event handler

Runs bebop verilator simulation:
  1. Resolve binary path
  2. Resolve verilog source directory (VSRC_PATH)
  3. Build bebop with verilator feature (cargo build)
  4. Run bebop verilator with elf, log and fst directories
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
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-verilator-sim",
    "description": "Run bebop verilator simulation",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.sim")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    arch_dir = f"{bbdir}/arch"

    arch_config = input_data.get("config", "sims.verilator.BuckyballToyVerilatorConfig")
    vsrc_dir = get_verilator_build_dir(bbdir, arch_config, input_data.get("vsrc_dir"))
    ctx.logger.info(f"Using verilog source directory: {vsrc_dir}")

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

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = input_data.get("log_dir") or f"{arch_dir}/log/{timestamp}-{binary_name}"
    fst_dir = input_data.get("fst_dir") or f"{arch_dir}/waveform/{timestamp}-{binary_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fst_dir, exist_ok=True)

    # ── Run bebop verilator ───────────────────────────────────────────────
    trace_flags = []
    for flag in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace"):
        if input_data.get(flag, False):
            trace_flags.append(f"--{flag}")
    trace_args = (" " + " ".join(trace_flags)) if trace_flags else ""

    run_cmd = (
        f"cargo run --features verilator "
        f"--config=\"env.ARCH_CONFIG='{arch_config}'\" "
        f"--config=\"env.VSRC_PATH='{vsrc_dir}'\" "
        f"-- verilator "
        f"--elf=\"{binary_path}\" "
        f"--log-dir=\"{log_dir}\" "
        f"--fst-dir=\"{fst_dir}\""
        f"{trace_args}"
    )
    ctx.logger.info(f"Running bebop verilator: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop verilator",
        stderr_prefix="bebop verilator",
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilator",
            "binary": binary_path,
            "config": arch_config,
            "log_dir": log_dir,
            "fst_dir": fst_dir,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
