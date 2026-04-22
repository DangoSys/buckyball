import os
import shlex
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
    "name": "dc-run",
    "description": "run Design Compiler synthesis script",
    "flows": ["dc"],
    "triggers": [queue("dc.run")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    script_path = f"{bbdir}/evals/run-dc.sh"

    srcdir = input_data.get("srcdir")
    top = input_data.get("top")
    keep_hierarchy = bool(input_data.get("keep_hierarchy", False))
    balltype = input_data.get("balltype")
    config_name = input_data.get("config", "sims.verilator.BuckyballToyVerilatorConfig")
    output_dir = input_data.get("output_dir")

    if not srcdir:
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "dc",
                "error": "Missing required parameter: srcdir",
                "example": "bbdev dc --srcdir arch/ReluBall_1 --top ReluBall",
            },
            trace_id=origin_tid,
        )
        return failure_result

    if not os.path.exists(script_path):
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "dc",
                "error": f"DC script not found: {script_path}",
            },
            trace_id=origin_tid,
        )
        return failure_result

    # Keep compatibility with old manual flow: ensure run-dc.sh is executable.
    os.chmod(script_path, os.stat(script_path).st_mode | 0o111)

    # Optional integrated pre-step: generate Ball verilog first.
    if balltype:
        final_output_dir = output_dir or f"{balltype}_1"
        verilog_dir = f"{bbdir}/arch/{final_output_dir}"
        verilog_cmd = (
            f"mill -i __.test.runMain sims.verify.BallTopMain {shlex.quote(str(balltype))} "
            "--disable-annotation-unknown --strip-debug-info -O=debug "
            f"--split-verilog -o={shlex.quote(verilog_dir)}"
        )

        verilog_result = stream_run_logger(
            cmd=verilog_cmd,
            logger=ctx.logger,
            cwd=f"{bbdir}/arch",
            stdout_prefix="dc verilog",
            stderr_prefix="dc verilog",
        )

        if verilog_result.returncode != 0:
            success_result, failure_result = await check_result(
                ctx,
                verilog_result.returncode,
                continue_run=False,
                extra_fields={
                    "task": "dc",
                    "error": "Failed to generate pre-DC verilog",
                    "balltype": balltype,
                    "config": config_name,
                },
                trace_id=origin_tid,
            )
            return failure_result

        # If user keeps default-like srcdir, force it to generated dir for consistency.
        srcdir = f"arch/{final_output_dir}"

    command_parts = ["bash", shlex.quote(script_path), "--srcdir", shlex.quote(str(srcdir))]
    if top:
        command_parts.extend(["--top", shlex.quote(str(top))])
    if keep_hierarchy:
        command_parts.append("--keep-hierarchy")

    command = " ".join(command_parts)

    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=bbdir,
        executable="bash",
        stdout_prefix="dc run",
        stderr_prefix="dc run",
    )

    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "dc",
            "report_dir": input_data.get("report_dir", f"{bbdir}/bb-tests/output/dc/reports"),
            "srcdir": srcdir,
            "top": top,
        },
        trace_id=origin_tid,
    )

    return
