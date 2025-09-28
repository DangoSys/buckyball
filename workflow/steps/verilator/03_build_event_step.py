import os
import subprocess
import glob
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
    "type": "event",
    "name": "make build",
    "description": "build verilator executable",
    "subscribes": ["verilator.build"],
    "emits": ["verilator.sim", "verilator.complete", "verilator.error"],
    "flows": ["verilator"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = f"{arch_dir}/build"
    waveform_dir = f"{arch_dir}/waveform"
    log_dir = f"{arch_dir}/log"

    # ==================================================================================
    # 执行操作
    # ==================================================================================
    # Find sources
    vsrcs = glob.glob(f"{build_dir}/**/*.v", recursive=True) + glob.glob(
        f"{build_dir}/**/*.sv", recursive=True
    )
    csrcs = (
        glob.glob(f"{arch_dir}/src/csrc/**/*.c", recursive=True)
        + glob.glob(f"{arch_dir}/src/csrc/**/*.cc", recursive=True)
        + glob.glob(f"{arch_dir}/src/csrc/**/*.cpp", recursive=True)
        + glob.glob(f"{build_dir}/**/*.c", recursive=True)
        + glob.glob(f"{build_dir}/**/*.cc", recursive=True)
        + glob.glob(f"{build_dir}/**/*.cpp", recursive=True)
    )

    # Setup paths
    inc_paths = [
        os.environ.get("RISCV", "") + "/include" if os.environ.get("RISCV") else "",
        f"{arch_dir}/thirdparty/chipyard/tools/DRAMSim2",
        build_dir,
        f"{arch_dir}/src/csrc/include",
    ]
    inc_flags = " ".join([f"-I{p}" for p in inc_paths if p])

    topname = "TestHarness"

    cflags = f"{inc_flags} -DTOP_NAME='\"V{topname}\"' -std=c++17 "
    ldflags = (
        f"-lreadline -ldramsim -lfesvr "
        f"-L{arch_dir}/thirdparty/chipyard/tools/DRAMSim2 "
        f"-L{arch_dir}/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build "
        f"-L{arch_dir}/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/lib"
    )

    obj_dir = f"{build_dir}/obj_dir"
    subprocess.run(f"rm -rf {obj_dir}", shell=True)
    os.makedirs(obj_dir, exist_ok=True)

    sources = " ".join(vsrcs + csrcs)
    jobs = data.get("jobs", "")

    verilator_cmd = (
        f"verilator -MMD --build -cc --trace -O3 --x-assign fast --x-initial fast --noassert -Wno-fatal "
        f"--trace-fst --trace-threads 1 --output-split 10000 --output-split-cfuncs 100 "
        f"--unroll-count 256 "
        f"-Wno-PINCONNECTEMPTY "
        f"-Wno-ASSIGNDLY "
        f"-Wno-DECLFILENAME "
        f"-Wno-UNUSED "
        f"-Wno-UNOPTFLAT "
        f"-Wno-BLKANDNBLK "
        f"-Wno-style "
        f"-Wall "
        f"--timing -j {jobs} +incdir+{build_dir} --top {topname} {sources} "
        f"-CFLAGS '{cflags}' -LDFLAGS '{ldflags}' --Mdir {obj_dir} --exe"
    )

    result = stream_run_logger(
        cmd=verilator_cmd,
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="verilator build",
        stderr_prefix="verilator build",
    )
    result = stream_run_logger(
        cmd=f"make -C {obj_dir} -f V{topname}.mk {obj_dir}/V{topname}",
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="verilator build",
        stderr_prefix="verilator build",
    )

    # ==================================================================================
    # 向API返回结果
    # ==================================================================================
    if data.get("from_run_workflow"):
        await context.state.set(context.trace_id, "processing", True)
    elif result.returncode != 0:
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
    if data.get("from_run_workflow"):
        await context.emit({"topic": "verilator.sim", "data": {**data, "task": "run"}})
    elif result.returncode == 0:
        await context.emit(
            {
                "topic": "verilator.complete",
                "data": {**data, "task": "build", "result": success_result},
            }
        )
    else:
        await context.emit(
            {
                "topic": "verilator.error",
                "data": {**data, "task": "build", "result": failure_result},
            }
        )

    return
