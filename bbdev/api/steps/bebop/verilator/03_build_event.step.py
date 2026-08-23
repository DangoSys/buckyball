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

from utils.build import install_bundle
from utils.chip import bebop_cargo, require_chip
from utils.path import bebop_cargo_env, bebop_target_dir, get_buckyball_path, get_verilator_build_dir
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

    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return

    vsrc_dir = get_verilator_build_dir(bbdir, chip, input_data.get("vsrc_dir"))
    ctx.logger.info(f"Using verilog source directory: {vsrc_dir}")

    if not os.path.isdir(vsrc_dir):
        ctx.logger.error(f"VSRC_PATH does not exist: {vsrc_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "vsrc_not_found",
                "source": "bebop.verilator.build",
                "vsrc_dir": vsrc_dir,
            },
            trace_id=origin_tid,
        )
        return

    diff = bool(input_data.get("diff", False))
    build_dir = bebop_dir
    manifest = bebop_cargo(bbdir)
    chip = input_data.get("chip")
    features = ["verilator"]
    if diff:
        if not isinstance(chip, str) or not chip:
            ctx.logger.error("Missing required parameter: --chip (required with --diff)")
            await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={"error": "missing_chip"},
                trace_id=origin_tid,
            )
            return
        try:
            install_bundle(bbdir, chip)
        except ValueError as error:
            ctx.logger.error(str(error))
            await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={"error": "chip_manifest_not_found", "detail": str(error)},
                trace_id=origin_tid,
            )
            return
        manifest = Path(bbdir) / "examples" / "chips" / chip / "generated" / "bebop" / "Cargo.toml"
        build_dir = manifest.parent
        if not manifest.is_file():
            ctx.logger.error(f"missing verilator bebop shim: {manifest}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "bebop_shim_not_found", "manifest": str(manifest)},
                trace_id=origin_tid,
            )
            return
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
                await check_result(
                    ctx,
                    env_result.returncode,
                    continue_run=False,
                    extra_fields={
                        "task": "environment",
                        "diff": True,
                        "dependency": dramsim_header,
                    },
                    trace_id=origin_tid,
                )
                return

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
    build_inner = (
        f"cd {shlex.quote(build_dir)} && "
        f"exec {cargo_build_cmd}"
    )
    build_cmd = f"nix develop -c sh -c {shlex.quote(build_inner)}"
    command_cwd = bbdir
    env = {**os.environ, "VSRC_PATH": vsrc_dir, **bebop_cargo_env(bbdir, chip)}
    ctx.logger.info(f"Building bebop verilator (diff={diff}) ...")
    build_result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=command_cwd,
        stdout_prefix="bebop verilator build",
        stderr_prefix="bebop verilator build",
        env=env,
    )

    bebop_bin = os.path.join(bebop_target_dir(bbdir, chip), "release", "bebop")
    target_dir = bebop_target_dir(bbdir, chip)
    if build_result.returncode == 0:
        try:
            write_build_marker(target_dir, chip, vsrc_dir, bebop_bin, diff=diff)
        except OSError as e:
            ctx.logger.error(f"failed to write bebop verilator build marker: {e}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "build_marker_write_failed", "detail": str(e)},
                trace_id=origin_tid,
            )
            return
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

    # Continue routing to sim if from run workflow
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
