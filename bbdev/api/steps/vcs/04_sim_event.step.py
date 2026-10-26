from utils.path import design_output_root
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
from utils.event_common import require_design
from utils.path import (
    get_buckyball_path,
    log_dir,
    bebop_bin_path,
)
from utils.search_workload import search_workload
from utils.stream_run import stream_run_logger_async

config = {
    "name": "vcs-sim",
    "description": "run a VCS BBSimHarness simulation",
    "flows": ["vcs"],
    "triggers": [queue("vcs.sim")],
    "enqueues": [],
}


def _resolve_binary(bbdir: str, design: str, binary: str, selection: str) -> str | None:
    if os.path.isfile(binary):
        return os.path.abspath(binary)
    return search_workload(
        str(Path(design_output_root(bbdir, design, selection=selection)) / "workloads"),
        binary,
    )


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        design = require_design(input_data)
    except ValueError as exc:
        await check_result(
            ctx, 1, extra_fields={"task": "sim", "error": str(exc)}, trace_id=origin_tid
        )
        return
    binary_name = input_data.get("binary")
    if not isinstance(binary_name, str) or not binary_name:
        await check_result(
            ctx,
            1,
            extra_fields={"task": "sim", "error": "missing binary"},
            trace_id=origin_tid,
        )
        return
    bbdir = get_buckyball_path()
    binary = _resolve_binary(bbdir, design, binary_name, input_data["target"])
    if binary is None:
        await check_result(
            ctx,
            1,
            extra_fields={"task": "sim", "error": f"binary not found: {binary_name}"},
            trace_id=origin_tid,
        )
        return
    artifact_dir = (
        Path(design_output_root(bbdir, design, selection=input_data["target"])) / "vcs"
    )
    simv = artifact_dir / "simv"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = Path(
        log_dir(
            bbdir,
            design,
            "tapeout",
            timestamp,
            "vcs",
            binary_name,
            selection=input_data["target"],
        )
    )
    run_log.mkdir(parents=True)
    command = [
        bebop_bin_path(bbdir, design, "vcs", selection=input_data["target"]),
        "--simv",
        str(simv),
        "--elf",
        binary,
        "--dram-configs",
        str(Path(bbdir) / "result/share/dramsim3/configs"),
        "--log-dir",
        str(run_log),
    ]
    wave = input_data.get("wave", False)
    if wave:
        command.extend(["--wave", "vpd" if wave is True else str(wave).lower()])
    result = await stream_run_logger_async(
        cmd=shlex.join(command),
        logger=ctx.logger,
        cwd=artifact_dir,
        stdout_prefix="vcs sim",
        stderr_prefix="vcs sim",
    )
    await check_result(
        ctx,
        result.returncode,
        extra_fields={
            "task": "sim",
            "simv": str(simv),
            "binary": binary,
            "log_dir": str(run_log),
        },
        trace_id=origin_tid,
    )
