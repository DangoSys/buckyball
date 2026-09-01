import json
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
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger_async
from tapeout import clock_period_ns, get_tapeout_contract, write_run_tcl

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
        contract = get_tapeout_contract(get_buckyball_path(), input_data.get("chip"), top_module)
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
        gman = input_data.get("generate_manifest")
        if not isinstance(gman, str) or not gman:
            rman = input_data.get("replace_manifest")
            if isinstance(rman, str) and os.path.isfile(rman):
                with open(rman, encoding="utf-8") as handle:
                    gman = json.load(handle).get("generate_manifest")
        if not isinstance(gman, str) or not os.path.isfile(gman):
            raise ValueError("missing generate_manifest with link_dbs")
        with open(gman, encoding="utf-8") as handle:
            link_dbs = json.load(handle).get("link_dbs")
        if not isinstance(link_dbs, list) or not link_dbs:
            raise ValueError(f"generate_manifest.link_dbs empty: {gman}")
        for path in link_dbs:
            if not isinstance(path, str) or not os.path.isfile(path):
                raise ValueError(f"sram db missing: {path}")
        tech = {
            "target_library": contract.target_library,
            "synthetic_library": contract.synthetic_library,
            "link_library": list(contract.link_library) + list(link_dbs),
            "max_cores": contract.max_cores,
        }
        run_config = write_run_tcl(
            os.path.join(analysis_dir, "run.tcl"),
            {
                "top": top_module,
                "source_list": source_list_path,
                "output_dir": output_dir,
                "report_dir": report_dir,
                "sdc": str(contract.constraints_sdc),
                **tech,
            },
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        await check_result(ctx, 1, continue_run=False, extra_fields={"task": "dc", "error": str(exc)}, trace_id=origin_tid)
        return
    script_path = contract.dc_script
    dc_log = os.path.join(analysis_dir, "dc_shell.log")
    ctx.logger.info(f"Running chip-owned DC synthesis for {contract.chip}, top {top_module}: {script_path}")
    result = await stream_run_logger_async(
        cmd=(
            f"set -o pipefail; dc_shell -f {shlex.quote(str(script_path))} "
            f"-x {shlex.quote('set RUN_CONFIG ' + '{' + str(run_config) + '}')} "
            f"2>&1 | tee {shlex.quote(dc_log)}"
        ),
        logger=ctx.logger,
        cwd=os.path.dirname(script_path),
        stdout_prefix="dc",
        stderr_prefix="dc",
    )
    extra_fields = {
        "task": "area",
        "top_module": top_module,
        "dc_script": str(script_path),
        "dc_output_dir": output_dir,
        "dc_report_dir": report_dir,
        "dc_log": dc_log,
        "run_config": str(run_config),
        "chip": contract.chip,
        "tapeout_dir": str(contract.root),
        "replace_manifest": input_data.get("replace_manifest"),
    }

    if result.returncode == 0 and input_data.get("from_regression_area_power"):
        reg_scripts = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "regression", "scripts")
        )
        if reg_scripts not in sys.path:
            sys.path.insert(0, reg_scripts)
        from dc_area import area_mm2_from_rpt
        from result import merge_metrics

        rpt = os.path.join(report_dir, "area.rpt")
        if not os.path.isfile(rpt):
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={**extra_fields, "error": f"missing area.rpt: {rpt}", "area_rpt": rpt},
                trace_id=origin_tid,
            )
            return
        try:
            with open(rpt, encoding="utf-8") as handle:
                area = area_mm2_from_rpt(handle.read())
        except ValueError as exc:
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={**extra_fields, "error": str(exc), "area_rpt": rpt},
                trace_id=origin_tid,
            )
            return
        freq = 1000.0 / clock_period_ns(contract)
        merge_metrics(get_buckyball_path(), area=area, freq=freq)
        extra_fields["area"] = area
        extra_fields["freq"] = freq
        await check_result(
            ctx, result.returncode, continue_run=False,
            extra_fields=extra_fields, trace_id=origin_tid,
        )
        return

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=extra_fields,
        trace_id=origin_tid,
    )
    if result.returncode == 0 and input_data.get("from_power_workflow"):
        await ctx.enqueue({"topic": "dc.sim", "data": {**input_data, "area_dir": output_dir, "chip": contract.chip, "_trace_id": origin_tid}})
