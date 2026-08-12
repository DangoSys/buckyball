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
from utils.stream_run import stream_run_logger
from tapeout import get_tapeout_contract, technology_settings, write_run_tcl

config = {
    "name": "dc-area",
    "description": "run Synopsys Design Compiler and area reports on prepared DC RTL",
    "flows": ["dc"],
    "triggers": [queue("dc.area")],
    "enqueues": ["dc.sim"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    source_list_path = input_data.get("source_list")
    top_module = input_data.get("top") or "DigitalTop"
    analysis_dir = input_data.get("analysis_dir")
    if not isinstance(source_list_path, str) or not os.path.isfile(source_list_path):
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "dc", "error": "missing prepared DC source list"},
            trace_id=origin_tid,
        )
        return
    if not isinstance(analysis_dir, str) or not analysis_dir:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "dc", "error": "missing fixed DC analysis output directory"},
            trace_id=origin_tid,
        )
        return
    if shutil.which("dc_shell") is None:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "dc",
                "error": "dc_shell is not on PATH; source the DC host environment before running bbdev dc --area",
            },
            trace_id=origin_tid,
        )
        return

    try:
        contract = get_tapeout_contract(get_buckyball_path(), input_data.get("config"), top_module)
    except (OSError, ValueError) as exc:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "dc", "error": str(exc)},
            trace_id=origin_tid,
        )
        return

    output_dir = os.path.join(analysis_dir, "outputs")
    report_dir = os.path.join(analysis_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    try:
        tech = technology_settings()
        run_config = write_run_tcl(
            os.path.join(analysis_dir, "run.tcl"),
            {
                "top": top_module,
                "source_list": source_list_path,
                "output_dir": output_dir,
                "report_dir": report_dir,
                "clock_port": contract.clock_port,
                "clock_period_ns": contract.clock_period_ns,
                **tech,
            },
        )
    except (OSError, ValueError) as exc:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "dc", "error": str(exc)}, trace_id=origin_tid)
        return
    script_path = contract.dc_script
    dc_log = os.path.join(analysis_dir, "dc_shell.log")
    ctx.logger.info(f"Running chip-owned DC synthesis for {contract.chip}, top {top_module}: {script_path}")
    result = stream_run_logger(
        cmd=(
            f"dc_shell -f {shlex.quote(str(script_path))} "
            f"-x {shlex.quote('set RUN_CONFIG ' + '{' + str(run_config) + '}')} "
            f"> {shlex.quote(dc_log)} 2>&1"
        ),
        logger=ctx.logger,
        cwd=os.path.dirname(script_path),
        stdout_prefix="dc",
        stderr_prefix="dc",
    )
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "area",
            "top_module": top_module,
            "dc_script": str(script_path),
            "dc_output_dir": output_dir,
            "dc_report_dir": report_dir,
            "dc_log": dc_log,
            "run_config": str(run_config),
            "chip": contract.chip,
            "tapeout_dir": str(contract.root),
            "sram_manifest": (input_data.get("sram_collateral") or {}).get("sram_manifest"),
        },
        trace_id=origin_tid,
    )
    if result.returncode == 0 and input_data.get("from_power_workflow"):
        await ctx.enqueue({"topic": "dc.sim", "data": {**input_data, "area_dir": output_dir, "chip": contract.chip, "_trace_id": origin_tid}})
