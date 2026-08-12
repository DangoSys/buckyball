import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path, get_vcs_build_dir
from utils.stream_run import stream_run_logger


config = {
    "name": "vcs-verilog",
    "description": "generate BBSimHarness RTL for VCS",
    "flows": ["vcs"],
    "triggers": [queue("vcs.verilog")],
    "enqueues": ["vcs.build"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    config_name = input_data.get("config")
    if not isinstance(config_name, str) or not config_name or config_name == "None":
        await check_result(ctx, 1, extra_fields={"task": "verilog", "error": "missing config"}, trace_id=origin_tid)
        return
    bbdir = get_buckyball_path()
    build_dir = get_vcs_build_dir(bbdir, config_name, input_data.get("output_dir"))
    command = (
        f"mill -i __.test.runMain sims.verilator.Elaborate {config_name} "
        "--disable-annotation-unknown --strip-debug-info -O=release "
        f"--split-verilog -o={build_dir}"
    )
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=f"{bbdir}/arch",
        stdout_prefix="vcs verilog",
        stderr_prefix="vcs verilog",
    )
    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog", "output_dir": build_dir},
        trace_id=origin_tid,
    )
    if result.returncode == 0 and input_data.get("from_run_workflow"):
        await ctx.enqueue({"topic": "vcs.build", "data": {**input_data, "output_dir": build_dir, "_trace_id": origin_tid}})
