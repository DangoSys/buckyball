import importlib.util
import os
import sys
import re
from pathlib import Path
from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from utils.path import get_buckyball_path, workload_build_dir
from utils.event_common import check_result, get_origin_trace_id

_stack_build_path = os.path.join(
    os.path.dirname(__file__), "..", "stack", "scripts", "build.py"
)
_spec = importlib.util.spec_from_file_location("stack_build_module", _stack_build_path)
stack_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stack_build)
config = {
    "name": "workload-build",
    "description": "build workload",
    "flows": ["workload"],
    "triggers": [queue("workload.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    allowed = {
        "target",
        "device",
        "core",
        "design",
        "ctest",
        "mlirtest",
        "soctest",
        "_trace_id",
    }
    unknown = sorted((k for k in input_data if k not in allowed))
    if unknown:
        ctx.logger.error(f"Unknown workload build parameter(s): {', '.join(unknown)}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "unknown_parameter", "parameters": unknown},
            trace_id=origin_tid,
        )
        return
    design = input_data.get("design")
    if not design:
        ctx.logger.error("Missing required parameter: design must be specified")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_design"},
            trace_id=origin_tid,
        )
        return
    if not isinstance(design, str) or not re.fullmatch("[A-Za-z0-9_-]+", design):
        ctx.logger.error(f"Invalid design: {design}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_design", "design": design},
            trace_id=origin_tid,
        )
        return
    design_dir = Path(bbdir) / "examples" / "designs" / design
    if not design_dir.is_dir():
        ctx.logger.error(f"Workload design does not exist: {design}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "unknown_design", "design": design},
            trace_id=origin_tid,
        )
        return
    ctest = input_data.get("ctest", False)
    mlirtest = input_data.get("mlirtest", False)
    soctest = input_data.get("soctest", False)
    if not all(isinstance(flag, bool) for flag in (ctest, mlirtest, soctest)):
        ctx.logger.error("--ctest, --mlirtest and --soctest must be boolean flags")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_test_scope"},
            trace_id=origin_tid,
        )
        return
    if sum((ctest, mlirtest, soctest)) > 1:
        ctx.logger.error("--ctest, --mlirtest and --soctest are mutually exclusive")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "conflicting_test_scope"},
            trace_id=origin_tid,
        )
        return
    try:
        # Workloads link the device runtime; MLIR tests compile with its plugins.
        design_cmake = stack_build.build_design(
            bbdir,
            design=design,
            selection=input_data["target"],
            logger=ctx.logger,
            task_scope=origin_tid,
        )
        build = workload_build_dir(bbdir, design, selection=input_data["target"])
        stack_build.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                os.path.join(bbdir, "bb-tests", "workloads"),
                "-B",
                build,
                f"-DDESIGN_CMAKE={design_cmake}",
            ],
            repo=Path(bbdir),
            logger=ctx.logger,
            task_scope=origin_tid,
            output_prefix="workload configure",
        )
        target = (
            "sync-ctest-bin"
            if ctest
            else (
                "sync-mlirtest-bin"
                if mlirtest
                else "sync-soctest-bin" if soctest else "sync-bin"
            )
        )
        stack_build.run(
            ["ninja", "-C", build, "-j", "1", target],
            repo=Path(bbdir),
            logger=ctx.logger,
            task_scope=origin_tid,
            output_prefix="workload build",
        )
    except Exception as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "error": "workload_build_failed",
                "design": design,
                "detail": str(error),
            },
            trace_id=origin_tid,
        )
        return
    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "design": design,
            "ctest": ctest,
            "mlirtest": mlirtest,
            "soctest": soctest,
        },
        trace_id=origin_tid,
    )
    return
