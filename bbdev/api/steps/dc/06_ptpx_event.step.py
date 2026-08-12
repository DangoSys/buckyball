import os
import shlex
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
step_path = os.path.dirname(__file__)
if step_path not in sys.path:
    sys.path.insert(0, step_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from power import read_dynamic_power
from utils.stream_run import stream_run_logger
from tapeout import get_tapeout_contract, resolve_power_window, technology_settings, write_run_tcl

config = {
    "name": "dc-ptpx",
    "description": "run PrimeTime PX power analysis on DC output",
    "flows": ["dc"],
    "triggers": [queue("dc.power")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    top_module = input_data.get("top") or "DigitalTop"
    activity_path = input_data.get("activity")
    activity_format = input_data.get("format")
    analysis_dir = input_data.get("analysis_dir")
    if not isinstance(analysis_dir, str) or not analysis_dir:
        error = "missing fixed DC analysis output directory"
    elif not isinstance(activity_path, str) or not activity_path:
        error = "missing required --activity"
    elif not isinstance(activity_format, str) or activity_format not in {"saif", "vcd", "fsdb"}:
        error = "missing or invalid --format; expected saif, vcd, or fsdb"
    elif shutil.which("pt_shell") is None:
        error = "pt_shell is not on PATH; source the DC host environment before running bbdev dc --power"
    else:
        error = None
    if error:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "power", "error": error}, trace_id=origin_tid)
        return

    try:
        netlist = os.path.join(analysis_dir, "outputs", f"{top_module}.v")
        sdc = os.path.join(analysis_dir, "outputs", f"{top_module}.sdc")
        contract = get_tapeout_contract(get_buckyball_path(), input_data.get("config"), top_module)
        start_ns, end_ns = resolve_power_window(
            contract,
            input_data.get("start_ns", input_data.get("start-ns")),
            input_data.get("end_ns", input_data.get("end-ns")),
        )
        tech = technology_settings()
        run_config = write_run_tcl(
            os.path.join(analysis_dir, "power-run.tcl"),
            {
                "top": top_module,
                "netlist": netlist,
                "sdc": sdc,
                "activity": activity_path,
                "activity_format": activity_format,
                "report_dir": os.path.join(analysis_dir, "power-reports"),
                "strip_path": input_data.get("strip_path") or "",
                "start_ns": start_ns or "",
                "end_ns": end_ns or "",
                **tech,
            },
        )
        script = contract.power_script
    except (OSError, ValueError) as exc:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "power", "error": str(exc)}, trace_id=origin_tid)
        return

    result = stream_run_logger(
        cmd=(
            f"pt_shell -f {shlex.quote(str(script))} "
            f"-x {shlex.quote('set RUN_CONFIG ' + '{' + str(run_config) + '}')} "
            f"> {shlex.quote(os.path.join(analysis_dir, 'pt_shell.log'))} 2>&1"
        ),
        logger=ctx.logger,
        cwd=os.path.dirname(script),
        stdout_prefix="ptpx",
        stderr_prefix="ptpx",
    )
    report_dir = os.path.join(analysis_dir, "power-reports")
    dynamic_power = read_dynamic_power(os.path.join(report_dir, "power_total.rpt"))
    extra_fields = {
        "task": "power",
        "top_module": top_module,
        "activity": activity_path,
        "format": activity_format,
        "power_script": str(script),
        "run_config": str(run_config),
        "pt_log": os.path.join(analysis_dir, "pt_shell.log"),
        "power_report_dir": report_dir,
        "power_scope": "dynamic (internal + switching)",
        "start_ns": start_ns,
        "end_ns": end_ns,
    }
    if dynamic_power:
        extra_fields["dynamic_power"] = dynamic_power
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=extra_fields,
        trace_id=origin_tid,
    )
