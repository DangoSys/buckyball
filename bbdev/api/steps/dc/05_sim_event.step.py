import os
import shlex
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
step_path = os.path.dirname(__file__)
if step_path not in sys.path:
    sys.path.insert(0, step_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.search_workload import search_workload
from utils.stream_run import stream_run_logger
from tapeout import get_tapeout_contract, resolve_power_window, write_run_env

config = {
    "name": "dc-sim",
    "description": "rerun chip-owned simulation and produce activity for PTPX",
    "flows": ["dc"],
    "triggers": [queue("dc.sim")],
    "enqueues": ["dc.power"],
}


def _first(data: dict, *names: str):
    for name in names:
        value = data.get(name)
        if value is not None and value != "":
            return value
    return None


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    config_name = input_data.get("config")
    top_module = input_data.get("top") or "DigitalTop"
    analysis_dir = input_data.get("analysis_dir")
    if not isinstance(analysis_dir, str) or not analysis_dir:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": "missing analysis directory"}, trace_id=origin_tid)
        return

    try:
        contract = get_tapeout_contract(get_buckyball_path(), config_name, top_module)
    except (OSError, ValueError) as exc:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": str(exc)}, trace_id=origin_tid)
        return

    activity_format = str(_first(input_data, "format") or contract.power_format).lower()
    if activity_format not in {"saif", "vcd", "fsdb"}:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": f"unsupported activity format: {activity_format}"}, trace_id=origin_tid)
        return

    # An explicit activity file is retained as a debugging escape hatch. Normal
    # power runs leave it unset and always regenerate activity below.
    explicit_activity = _first(input_data, "activity")
    if explicit_activity:
        if not os.path.isfile(explicit_activity):
            await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": f"activity file does not exist: {explicit_activity}"}, trace_id=origin_tid)
            return
        await ctx.enqueue({"topic": "dc.power", "data": {**input_data, "activity": explicit_activity, "format": activity_format, "_trace_id": origin_tid}})
        return

    if shutil.which("bash") is None:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": "bash is required for chip power_sim.sh"}, trace_id=origin_tid)
        return

    activity_dir = os.path.join(analysis_dir, "activity")
    os.makedirs(activity_dir, exist_ok=True)
    workload_name = _first(input_data, "workload") or contract.power_workload
    if not workload_name:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": "missing workload name; pass --workload <built ELF name>"}, trace_id=origin_tid)
        return
    if os.path.isabs(str(workload_name)):
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": "workload must be a built ELF name, not an absolute path"}, trace_id=origin_tid)
        return
    workload = None
    for root in ("output/workloads/src", "build/workloads/src"):
        workload = search_workload(os.path.join(get_buckyball_path(), "bb-tests", root), str(workload_name))
        if workload:
            break
    if not workload:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": f"workload not found in bb-tests output/build directories: {workload_name}"}, trace_id=origin_tid)
        return
    try:
        start_ns, end_ns = resolve_power_window(
            contract,
            _first(input_data, "start_ns", "start-ns"),
            _first(input_data, "end_ns", "end-ns"),
        )
    except ValueError as exc:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "sim", "error": str(exc)}, trace_id=origin_tid)
        return
    netlist = os.path.join(input_data.get("area_dir") or "", f"{top_module}.v")
    sdc = os.path.join(input_data.get("area_dir") or "", f"{top_module}.sdc")
    activity_path = os.path.join(activity_dir, f"activity.{activity_format}")
    run_env = write_run_env(
        os.path.join(activity_dir, "run.env"),
        {
            "OUTPUT_DIR": activity_dir,
            "ACTIVITY_FILE": activity_path,
            "ACTIVITY_FORMAT": activity_format,
            "TOP": top_module,
            "CONFIG": str(config_name or ""),
            "NETLIST": netlist,
            "SDC": sdc,
            "WORKLOAD": workload,
            "START_NS": start_ns,
            "END_NS": end_ns,
            "ACTIVITY_STRIP_PATH": contract.power_strip_path,
        },
    )

    ctx.logger.info(
        f"Running chip-owned power simulation for {contract.chip}; "
        f"format={activity_format}, workload={workload or '<chip default>'}, "
        f"window={start_ns or '<auto>'}..{end_ns or '<auto>'} ns"
    )
    result = stream_run_logger(
        cmd=f"bash {shlex.quote(str(contract.power_sim_script))} {shlex.quote(str(run_env))}",
        logger=ctx.logger,
        cwd=str(contract.root),
        stdout_prefix="power sim",
        stderr_prefix="power sim",
    )
    if result.returncode == 0 and not os.path.isfile(activity_path):
        result.returncode = 1
        error = f"chip power simulation completed without producing {activity_path}"
    else:
        error = None
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "sim",
            "chip": contract.chip,
            "power_sim_script": str(contract.power_sim_script),
            "activity": activity_path,
            "format": activity_format,
            "workload": workload,
            "start_ns": start_ns,
            "end_ns": end_ns,
            **({"error": error} if error else {}),
        },
        trace_id=origin_tid,
    )
    if result.returncode == 0:
        await ctx.enqueue({"topic": "dc.power", "data": {**input_data, "activity": activity_path, "format": activity_format, "strip_path": contract.power_strip_path, "_trace_id": origin_tid}})
