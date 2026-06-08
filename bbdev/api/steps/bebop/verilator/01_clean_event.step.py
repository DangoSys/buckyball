import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-verilator-clean",
    "description": "Clean verilator build directory",
    "flows": ["bebop"],
    "triggers": [
        queue("bebop.verilator.clean"),
        queue("bebop.verilator.run.clean"),
    ],
    "enqueues": ["bebop.verilator.verilog", "bebop.verilator.run.verilog"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    config_name = input_data.get("config", "sims.verilator.BuckyballToyVerilatorConfig")
    build_dir = get_verilator_build_dir(bbdir, config_name, input_data.get("output_dir"))

    command = f"rm -rf {build_dir}"
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop verilator clean",
        stderr_prefix="bebop verilator clean",
    )

    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "clean", "config": config_name, "output_dir": build_dir},
        trace_id=origin_tid,
    )

    # Continue routing to verilog if from run workflow
    if input_data.get("from_run_workflow"):
        data = {k: v for k, v in input_data.items() if k not in ("output_dir", "vsrc_dir")}
        if input_data.get("_explicit_output_dir"):
            data["output_dir"] = build_dir
        await ctx.enqueue(
            {"topic": "bebop.verilator.run.verilog", "data": {**data, "task": "run"}}
        )
