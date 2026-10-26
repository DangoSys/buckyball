"""
bebop verilator build event handler

Builds the entry's Verilator runner; bebop-verilator finds the entry's RTL by its output layout
"""

from utils.path import runner_manifest


import os
import shlex
import shutil
import sys
from pathlib import Path
from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
from utils.event_common import require_design
from utils.path import bebop_target_dir, bebop_bin_path, get_buckyball_path, rtl_dir
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from build_marker import write_build_marker

config = {
    "name": "bebop-verilator-build",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.build"), queue("bebop.verilator.run.build")],
    "enqueues": ["bebop.verilator.sim", "bebop.verilator.run.sim"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    design = require_design(input_data)
    vsrc_dir = rtl_dir(
        bbdir,
        design,
        "verilog",
        selection=input_data["target"],
    )
    ctx.logger.info(f"Using verilog source directory: {vsrc_dir}")
    diff = bool(input_data.get("diff", False))
    manifest = runner_manifest(bbdir, design, input_data["target"])
    build_dir = str(manifest.parent)
    features = ["verilator"]
    if diff:
        features.append("bemu")
    jobs = input_data.get("jobs", 16)
    cargo_build_cmd = shlex.join(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
            "--bin",
            "bebop-verilator",
            "--target-dir",
            bebop_target_dir(
                bbdir, design, "verilator", selection=input_data["target"]
            ),
            "--features",
            ",".join(features),
            "--jobs",
            str(jobs),
        ]
    )
    build_inner = f"cd {shlex.quote(str(build_dir))} && exec {cargo_build_cmd}"
    build_cmd = f"nix develop -c sh -c {shlex.quote(build_inner)}"
    ctx.logger.info(f"Building bebop verilator (diff={diff}) ...")
    build_result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop verilator build",
        stderr_prefix="bebop verilator build",
    )
    bebop_bin = bebop_bin_path(
        bbdir, design, "verilator", selection=input_data["target"]
    )
    target_dir = bebop_target_dir(
        bbdir, design, "verilator", selection=input_data["target"]
    )
    if build_result.returncode == 0:
        Path(bebop_bin).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(target_dir) / "release/bebop-verilator", bebop_bin)
        write_build_marker(target_dir, design, vsrc_dir, bebop_bin, diff=diff)
    await check_result(
        ctx,
        build_result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={
            "task": "build",
            "vsrc_dir": vsrc_dir,
            "binary": bebop_bin,
            "diff": diff,
            "design": design,
        },
        trace_id=origin_tid,
    )
    if build_result.returncode != 0:
        return
    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {
                "topic": "bebop.verilator.run.sim",
                "data": {
                    **input_data,
                    "design": design,
                    "task": "run",
                    "target": input_data["target"],
                },
            }
        )
