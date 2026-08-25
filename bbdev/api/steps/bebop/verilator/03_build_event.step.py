"""
bebop verilator build event handler

Builds bebop with verilator feature and VSRC_PATH
"""
import os
import shlex
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import require_chip
from utils.path import bebop_cargo_env, bebop_target_dir, get_buckyball_path, rtl_dir
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
    bebop_dir = f"{bbdir}/bebop"
    chip = require_chip(input_data)

    vsrc_dir = rtl_dir(bbdir, chip, "verilog", input_data.get("vsrc_dir"))
    ctx.logger.info(f"Using verilog source directory: {vsrc_dir}")

    diff = bool(input_data.get("diff", False))
    build_dir = bebop_dir
    manifest = Path(bbdir) / "bebop" / "Cargo.toml"
    features = ["verilator"]
    if diff:
        manifest = Path(bbdir) / "examples" / "chips" / chip / "generated" / "bebop" / "Cargo.toml"
        build_dir = manifest.parent
        features.extend(["bemu", "difftest"])
        dramsim_header = os.path.join(bbdir, "result", "include", "dramsim3.h")
        if not os.path.isfile(dramsim_header):
            ctx.logger.info(f"Nix environment is missing {dramsim_header}; rebuilding result")
            env_result = await stream_run_logger_async(
                cmd="nix build",
                logger=ctx.logger,
                cwd=bbdir,
                stdout_prefix="bebop verilator difftest environment",
                stderr_prefix="bebop verilator difftest environment",
            )
            if env_result.returncode != 0:
                raise RuntimeError(f"nix build failed: {env_result.returncode}")

    jobs = input_data.get("jobs", 16)
    cargo_build_cmd = shlex.join(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
            "--bin",
            "bebop",
            "--features",
            ",".join(features),
            "--jobs",
            str(jobs),
        ]
    )
    build_inner = f"cd {shlex.quote(build_dir)} && exec {cargo_build_cmd}"
    build_cmd = f"nix develop -c sh -c {shlex.quote(build_inner)}"
    env = {**os.environ, "VSRC_PATH": vsrc_dir, **bebop_cargo_env(bbdir, chip)}
    ctx.logger.info(f"Building bebop verilator (diff={diff}) ...")
    build_result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop verilator build",
        stderr_prefix="bebop verilator build",
        env=env,
    )

    bebop_bin = os.path.join(bebop_target_dir(bbdir, chip), "release", "bebop")
    target_dir = bebop_target_dir(bbdir, chip)
    if build_result.returncode == 0:
        write_build_marker(target_dir, chip, vsrc_dir, bebop_bin, diff=diff)
    await check_result(
        ctx,
        build_result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={
            "task": "build",
            "vsrc_dir": vsrc_dir,
            "binary": bebop_bin,
            "diff": diff,
            "chip": chip,
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
                    "vsrc_dir": vsrc_dir,
                    "chip": chip,
                    "task": "run",
                },
            }
        )
