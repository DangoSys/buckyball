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
from utils.event_common import require_chip
from utils.path import get_buckyball_path, log_dir, rtl_dir, workload_build_dir, workload_tests_root
from utils.search_workload import search_workload
from utils.stream_run import stream_run_logger_async


config = {
    "name": "vcs-sim",
    "description": "run a VCS BBSimHarness simulation",
    "flows": ["vcs"],
    "triggers": [queue("vcs.sim")],
    "enqueues": [],
}


def _resolve_binary(bbdir: str, chip: str, binary: str) -> str | None:
    if os.path.isfile(binary):
        return os.path.abspath(binary)
    for root in (
        workload_tests_root(bbdir, chip),
        os.path.join(workload_build_dir(bbdir, chip), "src"),
    ):
        found = search_workload(root, binary)
        if found:
            return found
    return None


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
    except ValueError as exc:
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": str(exc)}, trace_id=origin_tid)
        return
    binary_name = input_data.get("binary")
    if not isinstance(binary_name, str) or not binary_name:
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": "missing binary"}, trace_id=origin_tid)
        return
    bbdir = get_buckyball_path()
    binary = _resolve_binary(bbdir, chip, binary_name)
    if binary is None:
        await check_result(ctx, 1, extra_fields={"task": "sim", "error": f"binary not found: {binary_name}"}, trace_id=origin_tid)
        return
    build_dir = rtl_dir(bbdir, chip, "tapeout", input_data.get("output_dir"))
    artifact_dir = Path(build_dir) / "vcs"
    simv = artifact_dir / "simv"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = Path(log_dir(
        bbdir, chip, "tapeout", timestamp, "vcs", binary_name,
        input_data.get("output_dir"),
    ))
    run_log.mkdir(parents=True)
    timeout_ns = input_data.get("timeout_ns", input_data.get("timeout-ns", "100000000"))
    if float(timeout_ns) <= 0:
        raise ValueError("timeout-ns must be a positive number")
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
            f"+dramsim_ini_dir={shlex.quote(str(Path(bbdir) / 'result/share/dramsim3/configs'))}",
            f"+timeout-ns={shlex.quote(str(timeout_ns))}",
            "+permissive-off",
        ]
    )
    result = await stream_run_logger_async(
        cmd=command,
        logger=ctx.logger,
        cwd=artifact_dir,
        stdout_prefix="vcs sim",
        stderr_prefix="vcs sim",
    )
    await check_result(
        ctx,
        result.returncode,
        extra_fields={"task": "sim", "simv": str(simv), "binary": binary, "log_dir": str(run_log)},
        trace_id=origin_tid,
    )
