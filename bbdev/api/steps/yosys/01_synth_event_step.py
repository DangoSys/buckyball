import os
import subprocess
import glob
import sys
import yaml

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "yosys synth",
    "description": "run yosys synthesis for area estimation",
    "subscribes": ["yosys.synth"],
    "emits": [],
    "flows": ["yosys"],
}


def load_yosys_config():
    """Load yosys configuration from yaml file"""
    config_path = os.path.join(os.path.dirname(__file__), "scripts", "yosys-config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


async def handler(data, context):
    bbdir = get_buckyball_path()
    build_dir = data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"

    # Load yaml config as defaults, CLI args override
    yosys_cfg = load_yosys_config()
    top_module = data.get("top") or yosys_cfg.get("top") or "TargetBall"
    balltype = data.get("balltype") or yosys_cfg.get("balltype") or "vecball"
    liberty = yosys_cfg.get("liberty")  # Path to .lib file, configured in yosys-config.yaml only

    context.logger.info(f"Ball type: {balltype}, top module: {top_module}")

    # ==================================================================================
    # Step 1: Generate Verilog via BallTopMain (pure design, no DPI-C sim wrappers)
    # Using BallTopMain instead of Elaborate to avoid SimTSI/SimDRAM/SimJTAG files
    # that contain DPI-C imports which Yosys cannot parse
    # ==================================================================================
    context.logger.info("Step 1: Cleaning build directory and generating Verilog...")

    # Clean build directory first to remove stale files from previous Elaborate runs
    import shutil
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    verilog_command = (
        f"mill -i __.test.runMain sims.verify.BallTopMain {balltype} "
    )
    verilog_command += "--disable-annotation-unknown -strip-debug-info -O=debug "
    verilog_command += "-lowering-options=disallowLocalVariables "
    verilog_command += f"--split-verilog -o={build_dir}"

    result = stream_run_logger(
        cmd=verilog_command,
        logger=context.logger,
        cwd=arch_dir,
        stdout_prefix="yosys verilog",
        stderr_prefix="yosys verilog",
    )

    if result.returncode != 0:
        context.logger.error(f"Verilog generation failed with code {result.returncode}")
        success_result, failure_result = await check_result(
            context,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "verilog"},
        )
        return failure_result

    # Remove unwanted file (BallTopMain shouldn't generate this, but clean up just in case)
    for unwanted in ["TestHarness.sv", "TargetBall.sv"]:
        topname_file = f"{arch_dir}/{unwanted}"
        if os.path.exists(topname_file):
            os.remove(topname_file)

    # ==================================================================================
    # Step 2: Run Yosys synthesis for area estimation
    # ==================================================================================
    context.logger.info("Step 2: Running Yosys synthesis for area estimation...")

    # Collect all Verilog/SystemVerilog sources
    vsrcs = glob.glob(f"{build_dir}/**/*.v", recursive=True) + glob.glob(
        f"{build_dir}/**/*.sv", recursive=True
    )

    if not vsrcs:
        context.logger.error(f"No Verilog source files found in {build_dir}")
        success_result, failure_result = await check_result(
            context,
            1,
            continue_run=False,
            extra_fields={"task": "synth", "error": "No Verilog source files found"},
        )
        return failure_result

    # Create yosys log directory
    yosys_output_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(yosys_output_dir, exist_ok=True)

    # Build yosys read commands for all source files
    read_commands = "\n".join([f"read_verilog -sv {src}" for src in vsrcs])

    # Write yosys synthesis script
    yosys_script = f"{yosys_output_dir}/synth_area.ys"
    with open(yosys_script, "w") as f:
        f.write(f"{read_commands}\n")
        f.write(f"hierarchy -top {top_module}\n")
        f.write("proc\n")
        f.write("flatten\n")
        f.write("opt\n")
        f.write(f"synth -top {top_module}\n")

        if liberty and os.path.exists(liberty):
            # Map to standard cells using liberty file for accurate area
            context.logger.info(f"Using liberty file: {liberty}")
            f.write(f"dfflibmap -liberty {liberty}\n")
            f.write(f"abc -liberty {liberty}\n")
            f.write(f"stat -liberty {liberty}\n")
            f.write(f"tee -o {yosys_output_dir}/area_report.txt stat -liberty {liberty}\n")
            # Write mapped netlist for OpenSTA timing analysis
            f.write(f"write_verilog {yosys_output_dir}/synth_netlist.v\n")
        else:
            if liberty:
                context.logger.warn(f"Liberty file not found: {liberty}, falling back to generic stat")
            f.write("stat\n")
            f.write(f"tee -o {yosys_output_dir}/area_report.txt stat\n")

    context.logger.info(f"Yosys script written to {yosys_script}")
    context.logger.info(f"Synthesizing {len(vsrcs)} source files with top module: {top_module}")

    # Run yosys
    yosys_command = f"yosys -s {yosys_script}"

    result = stream_run_logger(
        cmd=yosys_command,
        logger=context.logger,
        cwd=build_dir,
        stdout_prefix="yosys synth",
        stderr_prefix="yosys synth",
    )

    # ==================================================================================
    # Step 3: Run OpenSTA timing analysis (only if liberty file is available)
    # ==================================================================================
    extra = {"task": "synth", "output_dir": yosys_output_dir}
    netlist_file = f"{yosys_output_dir}/synth_netlist.v"
    timing_report_file = f"{yosys_output_dir}/timing_report.txt"

    if liberty and os.path.exists(liberty) and os.path.exists(netlist_file) and result.returncode == 0:
        context.logger.info("Step 3: Running OpenSTA timing analysis...")

        # Get target clock period from config (default 10ns = 100MHz)
        clock_period = yosys_cfg.get("clock_period", 10.0)
        clock_name = yosys_cfg.get("clock_name", "clock")

        # Write OpenSTA TCL script
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
            logger=context.logger,
            cwd=yosys_output_dir,
            stdout_prefix="opensta",
            stderr_prefix="opensta",
        )

        if sta_result.returncode == 0 and os.path.exists(timing_report_file):
            with open(timing_report_file, "r") as f:
                extra["timing_report"] = f.read()
            context.logger.info(f"Timing report saved to {timing_report_file}")
        else:
            context.logger.warn(f"OpenSTA timing analysis failed (rc={sta_result.returncode})")
    elif result.returncode == 0:
        context.logger.info("Skipping OpenSTA: no liberty file configured or synthesis failed")

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    report_file = f"{yosys_output_dir}/area_report.txt"
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            extra["area_report"] = f.read()
        context.logger.info(f"Area report saved to {report_file}")

    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=False,
        extra_fields=extra,
    )

    return
