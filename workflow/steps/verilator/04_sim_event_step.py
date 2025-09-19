import os
import subprocess
import sys
from datetime import datetime

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.search_workload import search_workload

config = {
    "type": "event",
    "name": "make sim",
    "description": "run simulation",
    "subscribes": ["verilator.sim"],
    "emits": ["verilator.complete", "verilator.error"],
    "flows": ["verilator"],
}


async def handler(data, context):
    # ==================================================================================
    # 获取仿真参数
    # ==================================================================================
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = f"{arch_dir}/build"

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # 获取binary名字（去除路径）
    binary = data.get("binary", "")
    if not binary:
        failure_result = {
            "status": 400,
            "body": {
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is missing in event data",
                "timestamp": timestamp,
                "binary": binary,
            },
        }
        context.logger.error("binary参数缺失", failure_result)
        await context.state.set(context.trace_id, "failure", failure_result)
        await context.emit(
            {
                "topic": "verilator.error",
                "data": {**data, "task": "sim", "result": failure_result},
            }
        )
        return

    binary_name = data.get("binary", "")
    binary_path = search_workload(f"{bbdir}/bb-tests/output/workloads/src", binary_name)
    if not binary_path:
        failure_result = {
            "status": 400,
            "body": {
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary not found",
                "binary": binary_name,
            },
        }
        context.logger.error("在指定路径未找到对应的binary", failure_result)
        await context.state.set(context.trace_id, "failure", failure_result)
        await context.emit(
            {
                "topic": "verilator.error",
                "data": {**data, "task": "sim", "result": failure_result},
            }
        )
        return

    # 创建带时间戳和binary名字的日志目录
    log_dir = f"{arch_dir}/log/{timestamp}-{binary_name}"
    waveform_dir = f"{arch_dir}/waveform/{timestamp}-{binary_name}"
    topname = "TestHarness"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(waveform_dir, exist_ok=True)

    bin_path = f"{build_dir}/obj_dir/V{topname}"
    batch = data.get("batch", False)

    # 动态生成VCD文件路径
    vcd_path = f"{waveform_dir}/waveform.vcd"
    log_path = f"{log_dir}/bdb.log"

    # 清理旧的波形文件
    subprocess.run(f"rm -f {waveform_dir}/waveform.vcd", shell=True, check=True)
    # ==================================================================================
    # 执行仿真脚本，实现流式输出
    # ==================================================================================
    batch_param = "True" if batch else "False"
    sim_cmd = f"./scripts/sim.sh {bin_path} {binary_path} {log_dir}/stdout.log \
                {log_dir}/disasm.log {batch_param} {vcd_path} {log_path}"
    script_dir = os.path.dirname(__file__)

    result = stream_run_logger(
        cmd=sim_cmd,
        logger=context.logger,
        cwd=script_dir,
        stdout_prefix="verilator sim",
        stderr_prefix="verilator sim",
    )

    vcd2fst_cmd = f"vcd2fst {waveform_dir}/waveform.vcd {waveform_dir}/fstwaveform.fst"
    subprocess.run(vcd2fst_cmd, cwd=arch_dir, shell=True, check=True, text=True)
    # 清理旧的波形文件
    subprocess.run(f"rm -f {waveform_dir}/waveform.vcd", shell=True, check=True)

    # ==================================================================================
    # 返回仿真结果
    # ==================================================================================
    # 此处为run workflow的终点，status状态不再继续设为processing
    if result.returncode != 0:
        failure_result = {
            "status": 500,
            "body": {
                "success": False,
                "failure": True,
                "processing": False,
                "returncode": result.returncode,
                "binary": binary,
                # "stdout": result.stdout,
                # "stderr": result.stderr,
                "log_dir": log_dir,
                "waveform_dir": waveform_dir,
                "timestamp": timestamp,
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
                "log_dir": log_dir,
                "waveform_dir": waveform_dir,
                "binary": binary,
                "timestamp": timestamp,
                # "stdout": result.stdout,
                # "stderr": result.stderr,
            },
        }
        await context.state.set(context.trace_id, "success", success_result)

    # ==================================================================================
    #  finish workflow
    # ==================================================================================
    if result.returncode == 0:
        await context.emit(
            {
                "topic": "verilator.complete",
                "data": {**data, "task": "sim", "result": success_result},
            }
        )
    else:
        await context.emit(
            {
                "topic": "verilator.error",
                "data": {**data, "task": "sim", "result": failure_result},
            }
        )

    return
