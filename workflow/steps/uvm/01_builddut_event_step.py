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
    "name": "UVM Build DUT",
    "description": "build dut",
    "subscribes": ["uvm.builddut"],
    "emits": [],
    "flows": ["uvm"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    build_dir = f"{bbdir}/bb-tests/uvbb/dut/build"
    dut_dir = f"{bbdir}/bb-tests/uvbb/dut"
    arch_dir = f"{bbdir}/arch"
    # ==================================================================================
    # Execute operation
    # ==================================================================================
    command = f"cd {arch_dir} && mill -i __.uvbb.runMain uvbb.Elaborate "
    command += "--disable-annotation-unknown -strip-debug-info -O=debug "
    command += f"--split-verilog -o={build_dir}"
    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="uvm build dut",
        stderr_prefix="uvm build dut",
    )

    # Remove unwanted file
    topname_file = f"{arch_dir}/BallTop.sv"
    if os.path.exists(topname_file):
        os.remove(topname_file)

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context, result.returncode, continue_run=False
    )

    # ==================================================================================
    # Continue routing
    # Routing to verilog or finish workflow
    # For run workflow, continue to verilog; for standalone clean, complete
    # ==================================================================================

    return
