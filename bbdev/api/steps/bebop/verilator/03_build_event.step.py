"""
bebop verilator build event handler

Builds bebop with verilator feature and VSRC_PATH
"""
import os
import shlex
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
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

    arch_config = input_data.get("config")
    if not isinstance(arch_config, str) or not arch_config or arch_config == "None":
        ctx.logger.error("Missing required parameter: config must be specified")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_config"},
            trace_id=origin_tid,
        )
        return

    vsrc_dir = get_verilator_build_dir(bbdir, arch_config, input_data.get("vsrc_dir"))
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

    jobs = input_data.get("jobs", 16)
    build_cmd = (
        f"cargo build --release --bin bebop --features verilator --jobs {shlex.quote(str(jobs))}"
    )
    env = {**os.environ, "VSRC_PATH": vsrc_dir}
    ctx.logger.info("Building bebop verilator ...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop verilator build",
        stderr_prefix="bebop verilator build",
        env=env,
    )

    bebop_bin = f"{bebop_dir}/target/release/bebop"
    if build_result.returncode == 0:
        try:
            write_build_marker(bebop_dir, arch_config, vsrc_dir, bebop_bin)
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
            "config": arch_config,
            "vsrc_dir": vsrc_dir,
            "binary": bebop_bin,
        },
        trace_id=origin_tid,
    )
    if build_result.returncode != 0:
        return

    # Continue routing to sim if from run workflow
    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {"topic": "bebop.verilator.run.sim", "data": {**input_data, "vsrc_dir": vsrc_dir, "task": "run"}}
        )
