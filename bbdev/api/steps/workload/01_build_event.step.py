import importlib.util
import os
import sys
import re
from pathlib import Path
from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from utils.path import get_buckyball_path
from utils.event_common import check_result, get_origin_trace_id

_workload_build_path = os.path.join(
    get_buckyball_path(), "bb-tests", "workloads", "scripts", "build.py"
)
_spec = importlib.util.spec_from_file_location(
    "workload_build_module", _workload_build_path
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_workload_build_path}")
workload_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workload_build)
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
        "chip",
        "stable",
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
    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip must be specified")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    if not isinstance(chip, str) or not re.fullmatch("[A-Za-z0-9_-]+", chip):
        ctx.logger.error(f"Invalid chip: {chip}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return
    chip_dir = Path(bbdir) / "examples" / "chips" / chip
    if not chip_dir.is_dir():
        ctx.logger.error(f"Workload chip does not exist: {chip}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "unknown_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return
    stable = input_data.get("stable", False)
    if not isinstance(stable, bool):
        ctx.logger.error("Invalid parameter: stable must be a boolean flag")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_stable", "stable": stable},
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
        workload_build.build_workload(
            bbdir,
            chip,
            ctest=ctest,
            mlirtest=mlirtest,
            soctest=soctest,
            stable=stable,
            logger=ctx.logger,
            task_scope=origin_tid,
        )
    except Exception as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "error": "workload_build_failed",
                "chip": chip,
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
            "chip": chip,
            "ctest": ctest,
            "mlirtest": mlirtest,
            "soctest": soctest,
        },
        trace_id=origin_tid,
    )
    return
