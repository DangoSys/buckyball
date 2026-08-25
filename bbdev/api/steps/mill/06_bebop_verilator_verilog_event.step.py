"""
bebop verilator verilog event handler

Generates Verilog via mill for bebop verilator
"""
import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
mill_dir = os.path.dirname(__file__)
mill_scripts = os.path.join(mill_dir, "scripts")
if mill_dir not in sys.path:
    sys.path.insert(0, mill_dir)
if mill_scripts not in sys.path:
    sys.path.insert(0, mill_scripts)

from utils.event_common import require_chip
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger_async
from utils.path import rtl_out
import mill as mill_run
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-verilator-verilog",
    "description": "Generate verilog code via mill",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.verilog"), queue("bebop.verilator.run.verilog")],
    "enqueues": ["bebop.verilator.build", "bebop.verilator.run.build"],
}


def check_verilog_output(build_dir: str) -> dict:
    exists = os.path.exists(build_dir)
    is_dir = os.path.isdir(build_dir)
    sv_count = 0
    if is_dir:
        sv_count = sum(1 for name in os.listdir(build_dir) if name.endswith((".sv", ".v")))
    return {
        "exists": exists,
        "is_dir": is_dir,
        "sv_count": sv_count,
    }


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch = os.path.join(bbdir, "arch")
    chip = None
    try:
        chip = require_chip(input_data)
        mill_config, build_dir = rtl_out(
            bbdir, chip, "verilog", input_data.get("output_dir"),
        )
        os.makedirs(build_dir, exist_ok=True)
        ctx.logger.info(f"Using mill config: {mill_config}")
        ctx.logger.info(f"Using build directory: {build_dir}")
        returncode = (
            await stream_run_logger_async(
            cmd=mill_run.elaborate_cmd("sims.verilator.Elaborate", mill_config, build_dir),
            logger=ctx.logger,
            cwd=arch,
            stdout_prefix="bebop verilator verilog",
            stderr_prefix="bebop verilator verilog",
            )
        ).returncode
    except (ValueError, RuntimeError) as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": str(error),
                "example": 'bbdev bebop verilator --verilog "--chip toy"',
            },
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"Using chip: {chip}")

    output_status = check_verilog_output(build_dir)
    if returncode != 0:
        await check_result(
            ctx,
            returncode,
            continue_run=False,
            extra_fields={
                "task": "verilog",
                "chip": chip,
                "output_dir": build_dir,
                "output_status": output_status,
            },
            trace_id=origin_tid,
        )
        return

    if not output_status["is_dir"] or output_status["sv_count"] == 0:
        ctx.logger.error(
            f"Verilog output is invalid: {build_dir} "
            f"(exists={output_status['exists']}, is_dir={output_status['is_dir']}, "
            f"sv_count={output_status['sv_count']})"
        )
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "verilog",
                "error": "invalid_verilog_output",
                "chip": chip,
                "output_dir": build_dir,
                "output_status": output_status,
            },
            trace_id=origin_tid,
        )
        return

    await check_result(
        ctx,
        returncode,
        continue_run=input_data.get("from_run_workflow", False) and returncode == 0,
        extra_fields={"task": "verilog", "chip": chip, "output_dir": build_dir},
        trace_id=origin_tid,
    )

    if input_data.get("from_run_workflow") and returncode == 0:
        await ctx.enqueue(
            {"topic": "bebop.verilator.run.build", "data": {**input_data, "output_dir": build_dir, "vsrc_dir": build_dir, "task": "run"}}
        )
