import os
import re
import shlex
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.chip import available_compiler_chips, available_cores, resolve_chip_compiler_core, resolve_core
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "build-compiler",
    "description": "build compiler",
    "flows": ["compiler"],
    "triggers": [queue("compiler.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    stable = input_data.get("stable", False)
    if not isinstance(stable, bool):
        ctx.logger.error("Invalid parameter: stable must be a boolean flag")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_stable", "stable": stable},
            trace_id=origin_tid,
        )
        return

    chip = input_data.get("chip")
    core = input_data.get("core")
    if bool(chip) == bool(core):
        ctx.logger.error("Specify exactly one compiler target: chip or core")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_target_selection"},
            trace_id=origin_tid,
        )
        return
    target_name = core or chip
    if not isinstance(target_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", target_name):
        ctx.logger.error(f"Invalid compiler target: {target_name}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_target", "target": target_name},
            trace_id=origin_tid,
        )
        return

    try:
        core_package = (
            resolve_core(bbdir, core, require_compiler=True)
            if core
            else resolve_chip_compiler_core(bbdir, chip)
        )
    except ValueError as error:
        choices = available_cores(bbdir) if core else available_compiler_chips(bbdir)
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "unknown_core" if core else "unknown_chip",
                "core": core,
                "chip": chip,
                "available": choices,
            },
            trace_id=origin_tid,
        )
        return
    compiler_dir = core_package.compiler
    assert compiler_dir is not None

    buddy_dir = f"{bbdir}/compiler/thirdparty/buddy-mlir"
    build_dir = f"{buddy_dir}/build"
    llvm_build_dir = f"{buddy_dir}/llvm/build"

    def nix_cmd(args: list[str]) -> str:
        return shlex.join(["nix", "develop", "-c", *args])

    async def run_stage(stage: str, command: str):
        ctx.logger.info(f"Running compiler build stage '{stage}': {command}")
        result = stream_run_logger(
            cmd=command,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="compiler build",
            stderr_prefix="compiler build",
        )
        if result.returncode != 0:
            await check_result(
                ctx, result.returncode, continue_run=False,
                extra_fields={
                    "task": stage,
                    "chip": chip,
                    "core": core_package.name,
                    "stable": stable,
                },
                trace_id=origin_tid,
            )
            return False
        return True

    cmake_args = [
        "cmake", "-G", "Ninja",
        "-S", buddy_dir,
        "-B", build_dir,
        f"-DBUDDY_EXTERNAL_DIALECTS_DIR={compiler_dir}",
        f"-DMLIR_DIR={llvm_build_dir}/lib/cmake/mlir",
        f"-DLLVM_DIR={llvm_build_dir}/lib/cmake/llvm",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON",
        "-DPython3_EXECUTABLE=python3",
        "-DPython_EXECUTABLE=python",
    ]
    if not await run_stage("configure", nix_cmd(cmake_args)):
        return

    command = nix_cmd(["ninja", "-C", build_dir,
                       "buddy-opt", "buddy-translate", "buddy-llc"])
    mode = "stable" if stable else "custom"
    if not await run_stage(f"compiler-{core_package.name}-{mode}", command):
        return

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        ctx, 0, continue_run=False, trace_id=origin_tid)

    # ==================================================================================
    # Continue routing
    # ==================================================================================
    return
