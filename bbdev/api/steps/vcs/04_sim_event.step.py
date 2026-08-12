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
from utils.path import get_buckyball_path, get_vcs_build_dir
from utils.search_workload import search_workload
from utils.stream_run import stream_run_logger


config = {
    "name": "vcs-sim",
    "description": "run a VCS BBSimHarness simulation",
    "flows": ["vcs"],
    "triggers": [queue("vcs.sim")],
    "enqueues": [],
}


def _resolve_binary(bbdir: str, binary: str) -> str | None:
    if os.path.isfile(binary):
        return os.path.abspath(binary)
    for root in ("output/workloads/src", "build/workloads/src"):
        found = search_workload(f"{bbdir}/bb-tests/{root}", binary)
        if found:
            return found
    return None


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    config_name = input_data.get("config")
    binary_name = input_data.get("binary")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": "missing config"}, trace_id=origin_tid)
        return
    if not isinstance(binary_name, str) or not binary_name:
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": "missing binary"}, trace_id=origin_tid)
        return
    bbdir = get_buckyball_path()
    binary = _resolve_binary(bbdir, binary_name)
    if binary is None:
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": f"binary not found: {binary_name}"}, trace_id=origin_tid)
        return
    build_dir = get_vcs_build_dir(bbdir, config_name, input_data.get("output_dir"))
    artifact_dir = Path(build_dir) / "vcs"
    simv = artifact_dir / "simv"
    if not simv.is_file():
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": f"VCS executable missing: {simv}; run bbdev vcs --build first"}, trace_id=origin_tid)
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir = Path(bbdir) / "log" / f"{timestamp}-vcs-{config_name.replace('/', '_')}"
    log_dir.mkdir(parents=True)
    timeout_ns = input_data.get("timeout_ns", input_data.get("timeout-ns", "100000000"))
    try:
        if float(timeout_ns) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": "timeout-ns must be a positive number"}, trace_id=origin_tid)
        return
    wave = input_data.get("waveform", input_data.get("wave"))
    wave_arg = "+vcd" if str(wave).lower() == "vcd" else ("+vpd" if wave else "")
    command = " ".join(
        [
            "env -u LD_LIBRARY_PATH",
            shlex.quote(str(simv)),
            "+permissive",
            f"+elf={shlex.quote(binary)}",
            "+batch" if input_data.get("batch", False) else "",
            wave_arg,
            f"+timeout-ns={shlex.quote(str(timeout_ns))}",
            "+permissive-off",
        ]
    )
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=artifact_dir,
        stdout_prefix="vcs sim",
        stderr_prefix="vcs sim",
    )
    await check_result(
        ctx,
        result.returncode,
        extra_fields={"task": "sim", "simv": str(simv), "binary": binary, "log_dir": str(log_dir)},
        trace_id=origin_tid,
    )
