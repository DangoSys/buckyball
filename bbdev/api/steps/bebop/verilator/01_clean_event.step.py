import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir
from utils.stream_run import stream_run_logger_async
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
    build_dir = rtl_dir(
        bbdir,
        chip,
        "verilog",
        input_data.get("output_dir"),
        rushb=bool(input_data.get("rushB", False)),
    )

    command = f"rm -rf {build_dir}"
    result = await stream_run_logger_async(
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
        extra_fields={"task": "clean", "output_dir": build_dir},
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
