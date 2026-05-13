"""
bebop bemu event handler

Runs bebop bemu (Spike-based) emulator:
  1. Resolve binary (ELF) path
  2. Build bebop with bemu feature (cargo build)
  3. Run bebop bemu with the resolved ELF
"""
import os
import sys
from datetime import datetime

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-bemu-sim",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.sim")],
    "enqueues": [],
}


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

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = input_data.get("log_dir") or f"{arch_dir}/log/{timestamp}-{binary_name}-bemu"
    os.makedirs(log_dir, exist_ok=True)

    # ── Run bebop bemu ────────────────────────────────────────────────────
    run_cmd = (
        f"cargo run --features bemu -- bemu "
        f"--elf=\"{binary_path}\" "
        f"--log-dir=\"{log_dir}\""
    )
    ctx.logger.info(f"Running bebop bemu: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop bemu",
        stderr_prefix="bebop bemu",
    )

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
