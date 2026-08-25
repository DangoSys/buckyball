import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "uvm-ball-verilog",
    "description": "generate isolated ball verilog for UVM",
    "flows": ["uvm"],
    "triggers": [queue("uvm.verilog")],
    "enqueues": [],
}


def _balltype(data: dict) -> str:
    balltype = data.get("balltype")
    if not isinstance(balltype, str) or not balltype:
        raise ValueError("Missing required parameter: --balltype")
    return balltype


def _output_dir(data: dict) -> str:
    output_dir = data.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Missing required parameter: --output-dir")
    return output_dir


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    try:
        balltype = _balltype(input_data)
        build_dir = _output_dir(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": str(error),
                "example": 'bbdev uvm --verilog "--balltype ReluBall --output-dir <dir>"',
            },
            trace_id=origin_tid,
        )
        return failure_result

    os.makedirs(build_dir, exist_ok=True)
    command = (
        f"mill -i __.test.runMain sims.verify.BallTopMain {balltype} "
        "--disable-annotation-unknown --strip-debug-info -O=release "
        f"--split-verilog -o={build_dir} "
    )
    ctx.logger.info(f"Using build directory: {build_dir}")
    result = await stream_run_logger_async(
        cmd=command,
        logger=ctx.logger,
        cwd=f"{bbdir}/arch",
        stdout_prefix="uvm ball verilog",
        stderr_prefix="uvm ball verilog",
    )

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"task": "verilog", "balltype": balltype, "output_dir": build_dir},
        trace_id=origin_tid,
    )
