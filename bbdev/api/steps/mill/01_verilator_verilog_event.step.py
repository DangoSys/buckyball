import os
import subprocess
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-verilog",
    "description": "generate verilog code",
    "flows": ["verilator"],
    "triggers": [queue("verilator.verilog")],
    "enqueues": ["verilator.build"],
}


def prepare_verilator_verilog(build_dir: str, arch_dir: str, logger):
    unwanted_harness = f"{arch_dir}/BBSimHarness.sv"
    if os.path.exists(unwanted_harness):
        os.remove(unwanted_harness)

    for patch_file in [f"{build_dir}/mm.h", f"{build_dir}/mm.cc"]:
        if os.path.exists(patch_file):
            with open(patch_file, "r") as f:
                content = f.read()
            patched = "\n".join(
                line for line in content.splitlines()
                if "fesvr/memif.h" not in line and "fesvr/elfloader.h" not in line
            )
            if patched != content:
                with open(patch_file, "w") as f:
                    f.write(patched)
                logger.info(f"Patched fesvr includes from {patch_file}")


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"
    config_name = input_data.get("config")

    if not config_name or config_name == "None":
        ctx.logger.error("Configuration name is required but not provided")
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
            trace_id=origin_tid,
        )
        return failure_result

    ctx.logger.info(f"Using configuration: {config_name}")

    if input_data.get("balltype"):
        command = (
            f"mill -i __.test.runMain sims.verify.BallTopMain {input_data.get('balltype')} "
        )
    else:
        command = f"mill -i __.test.runMain sims.verilator.Elaborate {config_name} "

    command += "--disable-annotation-unknown "
    command += "--strip-debug-info "
    command += "-O=debug "
    command += f"--split-verilog -o={build_dir} "

    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="verilator verilog",
        stderr_prefix="verilator verilog",
    )

    prepare_verilator_verilog(build_dir, arch_dir, ctx.logger)

    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog"},
        trace_id=origin_tid,
    )

    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {"topic": "verilator.build", "data": {**input_data, "task": "run"}}
        )

    return
