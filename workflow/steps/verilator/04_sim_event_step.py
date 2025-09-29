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
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "make sim",
    "description": "run simulation",
    "subscribes": ["verilator.sim"],
    "emits": [],
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

    binary_name = data.get("binary", "")
    success_result, failure_result = await check_result(
        context, returncode=(binary_name == None), continue_run=True
    )

    binary_path = search_workload(f"{bbdir}/bb-tests/output/workloads/src", binary_name)
    success_result, failure_result = await check_result(
        context, returncode=(binary_path == None), continue_run=True
    )
    if failure_result:
        context.logger.error("binary not found", failure_result)
        return

    # Create log and waveform directory
    log_dir = f"{arch_dir}/log/{timestamp}-{binary_name}"
    waveform_dir = f"{arch_dir}/waveform/{timestamp}-{binary_name}"
    topname = "TestHarness"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(waveform_dir, exist_ok=True)

    bin_path = f"{build_dir}/obj_dir/V{topname}"
    batch = data.get("batch", False)

    # Create log and waveform file
    log_path = f"{log_dir}/bdb.log"
    fst_path = f"{waveform_dir}/waveform.fst"
    # Remove old waveform file
    subprocess.run(f"rm -f {waveform_dir}/waveform.vcd", shell=True, check=True)

    # ==================================================================================
    # 执行仿真脚本，实现流式输出
    # ==================================================================================
    # batch_param = "True" if batch else "False"
    # sim_cmd = f"./scripts/sim.sh {bin_path} {binary_path} {log_dir}/stdout.log \
    # {log_dir}/disasm.log {batch_param} {vcd_path} {log_path}"
    sim_cmd = (
        f"{bin_path} +permissive +loadmem={binary_path} +loadmem_addr=800000000 "
        f"{'+batch ' if batch else ''} "
        f"+fst={fst_path} +log={log_path} +permissive-off "
        f"{binary_path} > >(tee {log_dir}/stdout.log) 2> >(spike-dasm > {log_dir}/disasm.log)"
    )
    script_dir = os.path.dirname(__file__)

    result = stream_run_logger(
        cmd=sim_cmd,
        logger=context.logger,
        cwd=script_dir,
        stdout_prefix="verilator sim",
        stderr_prefix="verilator sim",
        executable="bash",
    )
    success_result, failure_result = await check_result(
        context, returncode=result.returncode, continue_run=True
    )
    if failure_result:
        context.logger.error("sim failed", failure_result)
        return

    if os.path.exists(f"{waveform_dir}/waveform.fst.heir"):
        subprocess.run(
            f"gtkwave -f {waveform_dir}/waveform.fst -H {waveform_dir}/waveform.fst.heir",
            shell=True,
            check=True,
        )

    # ==================================================================================
    # 返回仿真结果
    # ==================================================================================
    # 此处为run workflow的终点，status状态不再继续设为processing
    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "sim",
            "binary": binary_path,
            "log_dir": log_dir,
            "waveform_dir": waveform_dir,
            "timestamp": timestamp,
        },
    )

    # ==================================================================================
    #  finish workflow
    # ==================================================================================

    return
