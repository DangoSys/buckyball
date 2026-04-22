import glob
import os
import sys
import yaml

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "yosys synth",
    "description": "run yosys synthesis for area estimation",
    "flows": ["yosys"],
    "triggers": [queue("yosys.synth")],
    "enqueues": [],
}


def load_yosys_config():
    config_path = os.path.join(os.path.dirname(__file__), "scripts", "yosys-config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/")

    yosys_cfg = load_yosys_config()
    top_module = input_data.get("top") or yosys_cfg.get("top") or "BuckyballAccelerator"
    liberty = yosys_cfg.get("liberty")

    vsrcs = glob.glob(f"{build_dir}/**/*.sv", recursive=True)

    filtered_vsrcs = []
    skipped = []
    for f in vsrcs:
        with open(f, "r") as fh:
            content = fh.read()
        if "'{" in content or 'import "DPI-C"' in content:
            skipped.append(os.path.basename(f))
        else:
            filtered_vsrcs.append(f)
    vsrcs = filtered_vsrcs

    if skipped:
        ctx.logger.info(
            f"Skipped {len(skipped)} files with unsupported syntax: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}"
        )

    if not vsrcs:
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "synth", "error": "No Verilog source files found"},
            trace_id=origin_tid,
        )
        return failure_result

    yosys_output_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(yosys_output_dir, exist_ok=True)

    read_commands = "\n".join([f"read_verilog -sv {src}" for src in vsrcs])
    yosys_script = f"{yosys_output_dir}/synth_area.ys"
    with open(yosys_script, "w") as f:
        f.write(f"{read_commands}\n")
        f.write(f"hierarchy -top {top_module}\n")
        f.write("proc\n")
        f.write("opt\n")
        f.write(f"synth -top {top_module}\n")

        if liberty and os.path.exists(liberty):
            f.write(f"dfflibmap -liberty {liberty}\n")
            f.write(f"abc -liberty {liberty}\n")
            f.write(f"tee -o {yosys_output_dir}/hierarchy_report.txt stat -liberty {liberty}\n")
            f.write("flatten\n")
            f.write("opt\n")
            f.write(f"stat -liberty {liberty}\n")
            f.write(f"tee -o {yosys_output_dir}/area_report.txt stat -liberty {liberty}\n")
            f.write(f"write_verilog {yosys_output_dir}/synth_netlist.v\n")
        else:
            f.write(f"tee -o {yosys_output_dir}/hierarchy_report.txt stat\n")
            f.write("flatten\n")
            f.write("opt\n")
            f.write("stat\n")
            f.write(f"tee -o {yosys_output_dir}/area_report.txt stat\n")

    result = stream_run_logger(
        cmd=f"yosys -s {yosys_script}",
        logger=ctx.logger,
        cwd=build_dir,
        stdout_prefix="yosys synth",
        stderr_prefix="yosys synth",
    )

    extra = {"task": "synth", "output_dir": yosys_output_dir}
    netlist_file = f"{yosys_output_dir}/synth_netlist.v"
    timing_report_file = f"{yosys_output_dir}/timing_report.txt"

    if liberty and os.path.exists(liberty) and os.path.exists(netlist_file) and result.returncode == 0:
        clock_period = yosys_cfg.get("clock_period", 10.0)
        clock_name = yosys_cfg.get("clock_name", "clock")
        sta_script = f"{yosys_output_dir}/sta_timing.tcl"
        with open(sta_script, "w") as f:
            f.write(f"read_liberty {liberty}\n")
            f.write(f"read_verilog {netlist_file}\n")
            f.write(f"link_design {top_module}\n")
            f.write(f"create_clock [get_ports {clock_name}] -name clk -period {clock_period}\n")
            f.write(f"report_checks -path_delay max -format full > {timing_report_file}\n")
            f.write(f"report_checks -path_delay max -format full\n")
            f.write("exit\n")

        sta_result = stream_run_logger(
            cmd=f"sta {sta_script}",
            logger=ctx.logger,
            cwd=yosys_output_dir,
            stdout_prefix="opensta",
            stderr_prefix="opensta",
        )

        if sta_result.returncode == 0 and os.path.exists(timing_report_file):
            with open(timing_report_file, "r") as f:
                extra["timing_report"] = f.read()

    report_file = f"{yosys_output_dir}/area_report.txt"
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            extra["area_report"] = f.read()

    hierarchy_file = f"{yosys_output_dir}/hierarchy_report.txt"
    if os.path.exists(hierarchy_file):
        with open(hierarchy_file, "r") as f:
            extra["hierarchy_report"] = f.read()

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=extra,
        trace_id=origin_tid,
    )

    return
