"""
bebop verilator build event handler

Builds bebop with verilator feature and VSRC_PATH
"""
import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id
from utils.bebop_verilator import write_build_marker

config = {
    "name": "bebop-verilator-build",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.build"), queue("bebop.verilator.run.build")],
    "enqueues": ["bebop.verilator.sim", "bebop.verilator.run.sim"],
}


def describe_path(path: str) -> dict:
    info = {
        "cwd": os.getcwd(),
        "uid": os.getuid(),
        "exists": os.path.exists(path),
        "is_dir": os.path.isdir(path),
        "parent": os.path.dirname(path),
    }
    try:
        stat = os.stat(path)
        info.update({
            "mode": oct(stat.st_mode),
            "owner_uid": stat.st_uid,
            "owner_gid": stat.st_gid,
        })
    except OSError as e:
        info.update({"stat_error": str(e), "errno": e.errno})
    try:
        info["parent_entries"] = sorted(os.listdir(info["parent"]))[:20]
    except OSError as e:
        info["parent_error"] = str(e)
    return info


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    arch_config = input_data.get("config")
    if not arch_config:
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
        path_info = describe_path(vsrc_dir)
        ctx.logger.error(f"VSRC_PATH does not exist: {vsrc_dir}; path_info={path_info}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "vsrc_not_found",
                "source": "bebop.verilator.build",
                "vsrc_dir": vsrc_dir,
                "path_info": path_info,
            },
            trace_id=origin_tid,
        )
        return

    build_cmd = (
        f"cargo build --features verilator "
        f"--config=\"env.VSRC_PATH='{vsrc_dir}'\""
    )
    ctx.logger.info("Building bebop verilator ...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop verilator build",
        stderr_prefix="bebop verilator build",
    )

    bebop_bin = f"{bebop_dir}/target/debug/bebop"
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
