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
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "make build",
    "description": "build verilator executable",
    "subscribes": ["verilator.build"],
    "emits": ["verilator.sim"],
    "flows": ["verilator"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = f"{arch_dir}/build"
    coverage = data.get("coverage", False)

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

    # Patch fesvr includes out of build/mm.h and build/mm.cc.
    # These files are auto-copied from testchipip by Verilator as SimDRAM.v
    # companion sources. They reference fesvr/memif.h which we don't have
    # (fesvr is removed). The memif_t dependency was only used by SimDRAM.cc's
    # load_elf — our SimDRAM_bb.cc doesn't use it.
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
                context.logger.info(f"Patched fesvr includes from {patch_file}")

    topname = "BBSimHarness"

    # ==================================================================================
    # Build flags
    # ==================================================================================
    dramsim2_dir = f"{arch_dir}/thirdparty/chipyard/tools/DRAMSim2"

    # Find readline headers/libs in nix store (not in standard paths under nix)
    rl_headers = glob.glob("/nix/store/*readline*-dev/include/readline/readline.h")
    readline_inc = os.path.dirname(os.path.dirname(rl_headers[0])) if rl_headers else ""
    rl_libs = glob.glob("/nix/store/*readline*/lib/libreadline.so")
    readline_lib = os.path.dirname(rl_libs[0]) if rl_libs else ""

    # Find zlib headers/libs in nix store
    zlib_headers = glob.glob("/nix/store/*zlib*-dev/include/zlib.h")
    if not zlib_headers:
        zlib_headers = glob.glob("/nix/store/*zlib*/include/zlib.h")
    zlib_inc = os.path.dirname(zlib_headers[0]) if zlib_headers else ""
    zlib_libs = glob.glob("/nix/store/*zlib*/lib/libz.so")
    zlib_lib = os.path.dirname(zlib_libs[0]) if zlib_libs else ""

    inc_paths = [
        dramsim2_dir,
        build_dir,
        f"{arch_dir}/src/csrc/include",
    ]
    if readline_inc:
        inc_paths.append(readline_inc)
    if zlib_inc:
        inc_paths.append(zlib_inc)
    inc_flags = " ".join([f"-I{p}" for p in inc_paths if p])

    # -DBBSIM: selects VBBSimHarness in bdb.h / main.cc
    cflags = f"{inc_flags} -DBBSIM -DTOP_NAME='\"V{topname}\"' -std=c++17"

    ldflags = (
        f"-lreadline -ldramsim -lstdc++ -lz "
        f"-L{bbdir}/result/lib "
        f"-L{dramsim2_dir} "
    )
    if readline_lib:
        ldflags += f"-L{readline_lib} -Wl,-rpath,{readline_lib} "
    if zlib_lib:
        ldflags += f"-L{zlib_lib} -Wl,-rpath,{zlib_lib} "

    obj_dir = f"{build_dir}/obj_dir"
    subprocess.run(f"rm -rf {obj_dir}", shell=True)
    os.makedirs(obj_dir, exist_ok=True)

    sources = " ".join(vsrcs + csrcs)
    jobs = data.get("jobs", "")

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
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="verilator verilation",
        stderr_prefix="verilator verilation",
    )
    if result.returncode != 0:
        success_result, failure_result = await check_result(
            context, result.returncode, continue_run=False, extra_fields={"task": "build"}
        )
        return

    make_jobs = jobs if jobs else str(os.cpu_count() or 16)
    result = stream_run_logger(
        cmd=f"make -j{make_jobs} VM_PARALLEL_BUILDS=1 -C {obj_dir} -f V{topname}.mk V{topname}",
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="verilator build",
        stderr_prefix="verilator build",
    )

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=data.get("from_run_workflow", False),
        extra_fields={"task": "build"},
    )

    if data.get("from_run_workflow"):
        await context.emit(
            {"topic": "verilator.sim", "data": {**data, "task": "run"}}
        )

    return
