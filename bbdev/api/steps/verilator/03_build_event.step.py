import os
import subprocess
import glob
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-build",
    "description": "build verilator executable",
    "flows": ["verilator"],
    "triggers": [queue("verilator.build")],
    "enqueues": ["verilator.sim", "verilator.cosim"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = f"{arch_dir}/build"
    coverage = input_data.get("coverage", False)
    cosim = input_data.get("cosim", False)

    # ==================================================================================
    # Find sources
    # ==================================================================================
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

    # Exclude testchipip's SimDRAM.cc — our SimDRAM_bb.cc overrides memory_init
    csrcs = [f for f in csrcs if not f.endswith("SimDRAM.cc") or "src/csrc" in f]

    # Exclude testchipip's TSI/HTIF C++ sources (deleted in verilog step, but guard anyway).
    # tsi_tick DPI symbol is provided by tsi_stub.cc in src/csrc instead.
    _tsi_htif = {"testchip_tsi.cc", "testchip_htif.cc", "SimTSI.cc"}
    csrcs = [f for f in csrcs if os.path.basename(f) not in _tsi_htif]

    if cosim:
        topname = "ToyBuckyball"
    else:
        topname = "BBSimHarness"

    # ==================================================================================
    # Build flags
    # ==================================================================================
    result_dir = f"{bbdir}/result"

    def pkg_config(flag, pkg):
        r = subprocess.run(["pkg-config", flag, pkg], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    readline_inc = pkg_config("--variable=includedir", "readline")
    readline_lib = pkg_config("--variable=libdir", "readline")
    zlib_lib = pkg_config("--variable=libdir", "zlib")

    inc_flags = " ".join([
        f"-I{result_dir}/include",
        f"-I{build_dir}",
        f"-I{arch_dir}/src/csrc/include",
        f"-I{readline_inc}",
    ])

    # -DBBSIM: selects VBBSimHarness in bdb.h / main.cc
    # BDB NDJSON trace (+trace=...) is runtime-only; bbdev sim uses +trace=all (04_sim_event.step.py).
    cflags = f"{inc_flags} -DBBSIM -DTOP_NAME='\"V{topname}\"' -std=c++17"
    if cosim:
        cflags += " -DCOSIM"

    ldflags = (
        f"-lreadline -ldramsim -lstdc++ -lz "
        f"-L{result_dir}/lib "
        f"-L{readline_lib} -Wl,-rpath,{readline_lib} "
        f"-L{zlib_lib} -Wl,-rpath,{zlib_lib} "
    )

    obj_dir = f"{build_dir}/obj_dir"
    subprocess.run(f"rm -rf {obj_dir}", shell=True)
    os.makedirs(obj_dir, exist_ok=True)

    sources = " ".join(vsrcs + csrcs)
    jobs = input_data.get("jobs", "")

    # Fix nix runtime library paths
    libstdcpp_path = subprocess.run(
        "g++ -print-file-name=libstdc++.so", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if libstdcpp_path and "/" in libstdcpp_path:
        nix_lib_dir = os.path.dirname(os.path.realpath(libstdcpp_path))
        ldflags += f" -Wl,-rpath,{nix_lib_dir}"

    # Enable ccache if available
    if subprocess.run("command -v ccache", shell=True, capture_output=True).returncode == 0:
        os.environ["OBJCACHE"] = "ccache"

    # Use lld for faster linking if available
    use_lld = subprocess.run("command -v ld.lld", shell=True, capture_output=True).returncode == 0
    if use_lld:
        ldflags += " -fuse-ld=lld"

    # ==================================================================================
    # Run verilator
    # ==================================================================================
    verilator_cmd = (
        f"verilator -MMD -cc --vpi --trace -O3 --x-assign fast --x-initial fast --noassert -Wno-fatal "
        f"--trace-fst --trace-threads 1 --output-split 10000 --output-split-cfuncs 100 "
        f"--unroll-count 256 "
        f"{'--coverage-line ' if coverage else ''}"
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
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="verilator verilation",
        stderr_prefix="verilator verilation",
    )
    if result.returncode != 0:
        await check_result(
            ctx, result.returncode, continue_run=False, extra_fields={"task": "build"},
            trace_id=origin_tid,
        )
        return

    make_jobs = jobs if jobs else str(os.cpu_count() or 16)
    result = stream_run_logger(
        cmd=f"make -j{make_jobs} VM_PARALLEL_BUILDS=1 -C {obj_dir} -f V{topname}.mk V{topname}",
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="verilator build",
        stderr_prefix="verilator build",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "build"},
        trace_id=origin_tid,
    )

    # ==================================================================================
    # Continue routing
    # ==================================================================================
    if input_data.get("from_run_workflow"):
        if cosim:
            await ctx.enqueue(
                {"topic": "verilator.cosim", "data": {**input_data, "task": "run"}}
            )
        else:
            await ctx.enqueue(
                {"topic": "verilator.sim", "data": {**input_data, "task": "run"}}
            )

    return
