import os
import subprocess
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
    "type": "event",
    "name": "UVM Build DUT",
    "description": "build dut",
    "subscribes": ["uvm.builddut"],
    "emits": ["uvm.builddut", "uvm.builddut.complete", "uvm.builddut.error"],
    "flows": ["uvm"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    build_dir = f"{bbdir}/bb-tests/uvbb/dut/build"
    dut_dir = f"{bbdir}/bb-tests/uvbb/dut"
    arch_dir = f"{bbdir}/arch"
    # ==================================================================================
    # 执行操作
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
    # 向API返回结果
    # ==================================================================================
    if result.returncode != 0:
        failure_result = {
            "status": 500,
            "body": {
                "success": False,
                "failure": True,
                "processing": False,
                "returncode": result.returncode,
                # "stdout": result.stdout,
                # "stderr": result.stderr,
            },
        }
        await context.state.set(context.trace_id, "failure", failure_result)
    else:
        success_result = {
            "status": 200,
            "body": {
                "success": True,
                "failure": False,
                "processing": False,
                "returncode": result.returncode,
                # "stdout": result.stdout,
                # "stderr": result.stderr,
            },
        }
        await context.state.set(context.trace_id, "success", success_result)

    # ==================================================================================
    # 继续路由
    # Routing to verilog or finish workflow
    # For run workflow, continue to verilog; for standalone clean, complete
    # ==================================================================================

    if result.returncode == 0:
        await context.emit(
            {
                "topic": "uvm.builddut.complete",
                "data": {**data, "task": "builddut", "result": success_result},
            }
        )
    else:
        await context.emit(
            {
                "topic": "uvm.builddut.error",
                "data": {**data, "task": "builddut", "result": failure_result},
            }
        )

    return
