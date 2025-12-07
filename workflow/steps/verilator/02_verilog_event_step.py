import os
import subprocess
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "make verilog",
    "description": "generate verilog code",
    "subscribes": ["verilator.verilog"],
    "emits": ["verilator.build"],
    "flows": ["verilator"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    build_dir = data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"

    # Get config name, must be provided
    config_name = data.get("config")
    if not config_name or config_name == "None":
        context.logger.error("Configuration name is required but not provided")
        success_result, failure_result = await check_result(
            context,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
        )
        return failure_result

    context.logger.info(f"Using configuration: {config_name}")

    # ==================================================================================
    # Execute operation
    # ==================================================================================
    if data.get("balltype"):
        command = (
            f"mill -i __.test.runMain sims.verify.BallTopMain {data.get('balltype')} "
        )
    else:
        command = f"mill -i __.test.runMain sims.verilator.Elaborate {config_name} "

    command += "--disable-annotation-unknown -strip-debug-info -O=debug "
    command += f"--split-verilog -o={build_dir}"

    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=arch_dir,
        stdout_prefix="verilator verilog",
        stderr_prefix="verilator verilog",
    )

    # Remove unwanted file
    topname_file = f"{arch_dir}/TestHarness.sv"
    if os.path.exists(topname_file):
        os.remove(topname_file)

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=data.get("from_run_workflow", False),
        extra_fields={"task": "verilog"},
    )

    # ==================================================================================
    # Continue routing
    # Routing to verilog or finish workflow
    # For run workflow, continue to verilog; for standalone clean, complete
    # ==================================================================================
    if data.get("from_run_workflow"):
        await context.emit(
            {"topic": "verilator.build", "data": {**data, "task": "run"}}
        )

    return
