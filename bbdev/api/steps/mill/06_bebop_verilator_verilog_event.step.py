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

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-verilator-verilog",
    "description": "Generate verilog code via mill",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.verilog")],
    "enqueues": ["bebop.verilator.build"],
}


def prepare_verilator_verilog(build_dir: str, arch_dir: str, logger):
    """Remove unwanted harness and patch fesvr includes"""
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
    build_dir = get_verilator_build_dir(
        bbdir,
        input_data.get("config"),
        input_data.get("output_dir"),
    )
    arch_dir = f"{bbdir}/arch"
    config_name = input_data.get("config")

    if not config_name or config_name == "None":
        ctx.logger.error("Configuration name is required but not provided")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev bebop verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
            trace_id=origin_tid,
        )
        return

    ctx.logger.info(f"Using configuration: {config_name}")
    ctx.logger.info(f"Using build directory: {build_dir}")

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
        stdout_prefix="bebop verilator verilog",
        stderr_prefix="bebop verilator verilog",
    )

    prepare_verilator_verilog(build_dir, arch_dir, ctx.logger)

    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog", "config": config_name, "output_dir": build_dir},
        trace_id=origin_tid,
    )

    # Continue routing to build if from run workflow
    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {"topic": "bebop.verilator.build", "data": {**input_data, "output_dir": build_dir, "vsrc_dir": build_dir, "task": "run"}}
        )
