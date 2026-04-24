import os
import subprocess
import sys
from datetime import datetime

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-sim",
    "description": "run simulation",
    "flows": ["verilator"],
    "triggers": [queue("verilator.sim")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    # ==================================================================================
    # Get simulation parameters
    # ==================================================================================
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = f"{arch_dir}/build"

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    binary_name = input_data.get("binary", "")
    coverage = input_data.get("coverage", False)
    success_result, failure_result = await check_result(
        ctx, returncode=(binary_name is None), continue_run=True, trace_id=origin_tid,
    )

    binary_path = search_workload(f"{bbdir}/bb-tests/output/workloads/src", binary_name)
    ctx.logger.info(f"binary_path: {binary_path}")
    success_result, failure_result = await check_result(
        ctx, returncode=(binary_path is None), continue_run=True, trace_id=origin_tid,
    )
    if failure_result:
        ctx.logger.error("binary not found", failure_result)
        return

    topname = "BBSimHarness"
    log_dir = f"{arch_dir}/log/{timestamp}-{binary_name}"
    waveform_dir = f"{arch_dir}/waveform/{timestamp}-{binary_name}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(waveform_dir, exist_ok=True)

    coverage_flag = ""
    if coverage:
        coverage_dat_path = f"{log_dir}/coverage.dat"
        coverage_flag = f"+verilator+coverage+file+{coverage_dat_path}"

    bin_path = f"{build_dir}/obj_dir/V{topname}"
    batch = input_data.get("batch", False)

    log_path    = f"{log_dir}/bdb.ndjson"
    stdout_path = f"{log_dir}/stdout.log"
    meta_path   = f"{log_dir}/sim_meta.txt"
    fst_path    = f"{waveform_dir}/waveform.fst"

    # ==================================================================================
    # Execute simulation
    # BBSimHarness uses +elf= for ELF loading (via BBSimDRAM.cc / libelf)
    # No fesvr, no +loadmem_addr needed
    #
    # disasm.log: only stderr -> spike-dasm (Rocket commit printf is stderr here;
    # merging stdout with 2>&1 can break: full stdio buffering + non-DASM bytes).
    # BDB_SIM_META moves NDJSON banner to sim_meta.txt so it does not pollute disasm.
    # ==================================================================================
    ld_lib_path = (
        f"{bbdir}/result/lib:"
        f"{arch_dir}/thirdparty/chipyard/tools/DRAMSim2"
    )
    sim_cmd = (
        f"export LD_LIBRARY_PATH=\"{ld_lib_path}:$LD_LIBRARY_PATH\"; "
        f"export BDB_SIM_META=\"{meta_path}\"; "
        f"{bin_path} +permissive "
        f"+elf={binary_path} "
        f"{'+batch ' if batch else ''}"
        f"{coverage_flag + ' ' if coverage_flag else ''}"
        f"+fst={fst_path} +log={log_path} +stdout={stdout_path} +trace=all +permissive-off "
        f"{binary_path} 2> >(spike-dasm > {log_dir}/disasm.log)"
    )
    script_dir = os.path.dirname(__file__)

    result = stream_run_logger(
        cmd=sim_cmd,
        logger=ctx.logger,
        cwd=script_dir,
        stdout_prefix="verilator sim",
        stderr_prefix="verilator sim",
        executable="bash",
    )
    success_result, failure_result = await check_result(
        ctx, returncode=result.returncode, continue_run=True, trace_id=origin_tid,
    )
    if failure_result:
        ctx.logger.error("sim failed", failure_result)
        return

    # ==================================================================================
    # Return simulation result
    # ==================================================================================
    extra_fields = {
        "task": "sim",
        "binary": binary_path,
        "log_dir": log_dir,
        "waveform_dir": waveform_dir,
        "timestamp": timestamp,
        "sim_meta": meta_path,
    }
    if coverage:
        extra_fields["coverage_dat"] = coverage_dat_path

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=extra_fields,
        trace_id=origin_tid,
    )

    return
