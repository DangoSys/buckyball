from pathlib import Path
import shlex
from datetime import datetime

from motia import FlowContext, queue
from utils.event_common import check_result, get_origin_trace_id, require_design
from utils.path import (
    get_buckyball_path,
    design_output_root,
    bebop_target_dir,
    runner_manifest,
    log_dir,
)
from utils.workload_manifest import resolve_workload_toml
from utils.stream_run import stream_run_logger_async

config = {
    "name": "bebop-vcs-batch",
    "flows": ["bebop"],
    "triggers": [queue("bebop.vcs.batch")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    design = require_design(input_data)
    repo = get_buckyball_path()
    selection = input_data["target"]
    tests = input_data.get("test", "bare-tests")
    manifest = resolve_workload_toml(design, "vcs", tests, repo, selection=selection)
    output = Path(design_output_root(repo, design, selection=selection))
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    report = (
        Path(
            log_dir(
                repo, design, "tapeout", stamp, "bebop-vcs", tests, selection=selection
            )
        )
        / origin_tid
    )
    command = [
        "cargo",
        "test",
        "--release",
        "--manifest-path",
        str(runner_manifest(repo, design, selection)),
        "--target-dir",
        bebop_target_dir(repo, design, "vcs", selection=selection),
        "--test",
        "test_vcs",
        "--",
        "--workload-toml",
        manifest,
        "--bb-tests-root",
        str(output / "workloads"),
        "--simv",
        str(output / "vcs/simv"),
        "--dram-configs",
        str(Path(repo) / "result/share/dramsim3/configs"),
        "--log-dir",
        str(report),
        "--jobs",
        str(input_data.get("jobs", 1)),
    ]
    result = await stream_run_logger_async(
        cmd=shlex.join(command),
        logger=ctx.logger,
        cwd=repo,
        stdout_prefix="bebop vcs batch",
        stderr_prefix="bebop vcs batch",
    )
    await check_result(
        ctx,
        result.returncode,
        extra_fields={
            "task": "batch",
            "backend": "vcs",
            "log_dir": str(report),
            "junit": str(report / "junit.xml"),
        },
        trace_id=origin_tid,
    )
