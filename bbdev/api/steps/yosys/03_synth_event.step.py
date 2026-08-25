import os
import re
import sys
import yaml
from datetime import datetime

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.path import rtl_dir, get_buckyball_path, log_dir
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "yosys synth",
    "description": "run yosys synthesis for area estimation",
    "flows": ["yosys"],
    "triggers": [queue("yosys.synth")],
    "enqueues": [],
}


POWER_TOTAL_RE = re.compile(
    r"^Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
    re.M,
)


def dynamic_power_report(power_report: str, activity_source: str) -> tuple[str, float] | None:
    """Return OpenSTA's internal-plus-switching power, excluding leakage."""
    match = POWER_TOTAL_RE.search(power_report)
    if match is None:
        return None
    internal, switching, _leakage, _total = (float(value) for value in match.groups())
    dynamic = internal + switching
    return (
        "OpenSTA dynamic power\n"
        f"  Activity source: {activity_source}\n"
        f"  Internal: {internal:.6e} W\n"
        f"  Switching: {switching:.6e} W\n"
        f"  Dynamic (internal + switching): {dynamic:.6e} W\n",
        dynamic,
    )


def load_yosys_config():
    config_path = os.path.join(os.path.dirname(__file__), "scripts", "yosys-config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            # Keep paths portable between the Nix shell and explicit user
            # configurations while retaining normal YAML scalar values.
            if isinstance(config.get("liberty"), str):
                config["liberty"] = os.path.expandvars(os.path.expanduser(config["liberty"]))
            if isinstance(config.get("vcd"), str) and config["vcd"]:
                config["vcd"] = os.path.expandvars(os.path.expanduser(config["vcd"]))
            return config
    return {}

async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    try:
        chip = require_chip(input_data)
        build_dir = rtl_dir(bbdir, chip, "synth", input_data.get("output_dir"))
    except ValueError as exc:
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"task": "synth", "error": str(exc)},
            trace_id=origin_tid,
        )
        return

    yosys_cfg = load_yosys_config()
    top_module = input_data.get("top") or yosys_cfg.get("top") or "DigitalTop"
    liberty = yosys_cfg.get("liberty") or os.environ.get("YOSYS_LIBERTY")
    if isinstance(liberty, str):
        liberty = os.path.expandvars(os.path.expanduser(liberty))

    source_list_path = input_data.get("source_list") or os.path.join(build_dir, "yosys_sources.list")
    if not os.path.exists(source_list_path):
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "synth", "error": "missing yosys_sources.list, run yosys verilog first"},
            trace_id=origin_tid,
        )
        return failure_result

    with open(source_list_path, "r") as f:
        vsrcs = [line.strip() for line in f.readlines() if line.strip()]

    if not vsrcs:
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "synth", "error": "empty yosys_sources.list"},
            trace_id=origin_tid,
        )
        return failure_result

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    yosys_output_dir = input_data.get("log_dir") or log_dir(
        bbdir, chip, "synth", stamp, "yosys", top_module, input_data.get("output_dir"),
    )
    os.makedirs(yosys_output_dir, exist_ok=True)
    ctx.logger.info(f"Yosys log dir: {yosys_output_dir}")

    sram_collateral = input_data.get("sram_collateral") or {}

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

    result = await stream_run_logger_async(
        cmd=f"yosys -s {yosys_script}",
        logger=ctx.logger,
        cwd=build_dir,
        stdout_prefix="yosys synth",
        stderr_prefix="yosys synth",
    )

    extra = {
        "task": "synth",
        "output_dir": yosys_output_dir,
        "sram_manifest": sram_collateral.get("sram_manifest"),
        "sram_memory_count": sram_collateral.get("sram_memory_count", 0),
    }
    netlist_file = f"{yosys_output_dir}/synth_netlist.v"
    timing_report_file = f"{yosys_output_dir}/timing_report.txt"
    power_report_file = f"{yosys_output_dir}/power_report.txt"
    dynamic_power_report_file = f"{yosys_output_dir}/dynamic_power_report.txt"
    sta_returncode = 0

    if liberty and os.path.exists(liberty) and os.path.exists(netlist_file) and result.returncode == 0:
        clock_period = yosys_cfg.get("clock_period", 10.0)
        clock_name = yosys_cfg.get("clock_name", "clock")
        vcd = input_data.get("vcd") or yosys_cfg.get("vcd")
        if isinstance(vcd, str):
            vcd = os.path.expandvars(os.path.expanduser(vcd))
        if vcd and not os.path.isfile(vcd):
            extra["error"] = f"VCD file does not exist: {vcd}"
            sta_returncode = 1
            vcd = None
        if sta_returncode:
            await check_result(
                ctx,
                sta_returncode,
                continue_run=False,
                extra_fields=extra,
                trace_id=origin_tid,
            )
            return
        sta_script = f"{yosys_output_dir}/sta_timing.tcl"
        with open(sta_script, "w") as f:
            f.write(f"read_liberty {liberty}\n")
            f.write(f"read_verilog {netlist_file}\n")
            f.write(f"link_design {top_module}\n")
            # Prefer the configured port, then use the generated-memory clock
            # naming convention.  Some small modules are combinational and
            # therefore have no clock at all.
            f.write(f"set bb_clock_port [get_ports -quiet {clock_name}]\n")
            f.write("if {[llength $bb_clock_port] == 0} { set bb_clock_port [get_ports -quiet *clk*] }\n")
            f.write(f"if {{[llength $bb_clock_port] > 0}} {{ create_clock $bb_clock_port -name clk -period {clock_period} }}\n")
            f.write(f"report_checks -path_delay max -format full > {timing_report_file}\n")
            f.write(f"report_checks -path_delay max -format full\n")
            if vcd and os.path.exists(vcd):
                f.write(f"read_vcd {vcd} -scope {top_module}\n")
                extra["power_activity_source"] = vcd
            else:
                input_activity = yosys_cfg.get("input_activity", 0.1)
                input_duty = yosys_cfg.get("input_duty", 0.5)
                clock_activity = yosys_cfg.get("clock_activity", 1.0)
                clock_duty = yosys_cfg.get("clock_duty", 0.5)
                f.write(f"set_power_activity -input -activity {input_activity} -duty {input_duty}\n")
                # OpenSTA expects the clock collection after ``-clock``.
                # Applying the default rate to all clocks keeps this usable
                # for generated designs whose clock port is renamed.
                f.write(f"if {{[llength [all_clocks]] > 0}} {{ set_power_activity -clock [all_clocks] -activity {clock_activity} -duty {clock_duty} }}\n")
                extra["power_activity_source"] = "default"
            f.write(f"report_power -digits 4 > {power_report_file}\n")
            f.write("exit\n")

        sta_result = await stream_run_logger_async(
            cmd=f"sta {sta_script}",
            logger=ctx.logger,
            cwd=yosys_output_dir,
            stdout_prefix="opensta",
            stderr_prefix="opensta",
        )

        if sta_result.returncode == 0 and os.path.exists(timing_report_file):
            with open(timing_report_file, "r") as f:
                extra["timing_report"] = f.read()
        sta_returncode = sta_result.returncode
        if sta_result.returncode == 0 and os.path.exists(power_report_file):
            with open(power_report_file, "r") as f:
                extra["power_report"] = f.read()
            dynamic_result = dynamic_power_report(
                extra["power_report"], extra.get("power_activity_source", "unknown")
            )
            if dynamic_result is not None:
                dynamic_report, dynamic_power = dynamic_result
                with open(dynamic_power_report_file, "w") as f:
                    f.write(dynamic_report)
                extra["dynamic_power_report"] = dynamic_report
                extra["dynamic_power_w"] = dynamic_power

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
        result.returncode or sta_returncode,
        continue_run=False,
        extra_fields=extra,
        trace_id=origin_tid,
    )

    return
