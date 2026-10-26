from pathlib import Path
import shlex
import shutil

from motia import FlowContext, queue
from utils.event_common import check_result, get_origin_trace_id, require_design
from utils.path import (
    get_buckyball_path,
    rtl_dir,
    design_output_root,
    runner_manifest,
    bebop_target_dir,
    bebop_bin_path,
)
from utils.stream_run import stream_run_logger_async

config = {
    "name": "vcs-build",
    "description": "Build the independent Bebop VCS node and SoC simulator",
    "flows": ["vcs"],
    "triggers": [queue("vcs.build")],
    "enqueues": ["vcs.sim"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    design = require_design(input_data)
    repo = get_buckyball_path()
    selection = input_data["target"]
    target = Path(bebop_target_dir(repo, design, "vcs", selection=selection))
    output = Path(design_output_root(repo, design, selection=selection)) / "vcs"
    rtl = rtl_dir(repo, design, "tapeout", selection=selection)
    commands = [
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(runner_manifest(repo, design, selection)),
            "--target-dir",
            str(target),
            "--bin",
            "bebop",
            "--bin",
            "bebop-vcs",
        ],
        [
            str(target / "release/bebop"),
            "build",
            "vcs",
            "--rtl-dir",
            rtl,
            "--artifact-dir",
            str(output),
            "--jobs",
            str(input_data.get("jobs", 16)),
        ],
    ]
    for command in commands:
        result = await stream_run_logger_async(
            cmd=shlex.join(command),
            logger=ctx.logger,
            cwd=repo,
            stdout_prefix="vcs build",
            stderr_prefix="vcs build",
        )
        if result.returncode:
            await check_result(
                ctx,
                result.returncode,
                extra_fields={"task": "build"},
                trace_id=origin_tid,
            )
            return
    installed = Path(bebop_bin_path(repo, design, "vcs", selection=selection))
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release/bebop-vcs", installed)
    await check_result(
        ctx,
        0,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={
            "task": "build",
            "simv": str(output / "simv"),
            "compile_log": str(output / "vcs_compile.log"),
        },
        trace_id=origin_tid,
    )
    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {"topic": "vcs.sim", "data": {**input_data, "_trace_id": origin_tid}}
        )
