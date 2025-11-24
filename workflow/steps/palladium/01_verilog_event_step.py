import os
import subprocess
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "make verilog",
    "description": "generate verilog code",
    "subscribes": ["palladium.verilog"],
    "emits": ["palladium.build"],
    "flows": ["palladium"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    # Use arch/build as the base directory for chipyard.Generator
    base_build_dir = f"{data.get('output_dir', f'{bbdir}/arch/build')}/palladium"
    # Output directory for final Verilog files
    verilog_output_dir = f"{base_build_dir}/verilog"
    arch_dir = f"{bbdir}/arch"

    # Get config name, must be provided
    config_name = data.get("config")
    if not config_name or config_name == "None":
        context.logger.error("Configuration name is required but not provided")
        success_result, failure_result = await check_result(
            context,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": "Configuration name is required. Please specify --config_name parameter.",
                "example": './bbdev palladium --verilog "--config_name sims.palladium.BuckyballToyP2EConfig"',
            },
        )
        return failure_result

    context.logger.info(f"Using configuration: {config_name}")

    # ==================================================================================
    # Step 1: Generate FIRRTL using chipyard.Generator
    # ==================================================================================
    context.logger.info("Step 1: Generating FIRRTL with chipyard.Generator...")
    os.system(f"mkdir -p {verilog_output_dir}")
    firrtl_command = (
        f"cd {arch_dir} && "
        f"sbt -J-Xmx256G -J-Xss64M -J-XX:+UseG1GC -J-XX:MaxGCPauseMillis=1000 "
        f'"runMain chipyard.Generator '
        f"-td {base_build_dir} "
        f"-T palladium.fpga.VCU118FPGATestHarness "
        f'-C {config_name}"'
    )

    result = stream_run_logger(
        cmd=firrtl_command,
        logger=context.logger,
        cwd=arch_dir,
        stdout_prefix="palladium firrtl",
        stderr_prefix="palladium firrtl",
    )

    if result.returncode != 0:
        context.logger.error(f"FIRRTL generation failed with code {result.returncode}")
        success_result, failure_result = await check_result(
            context,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "firrtl", "step": "generate"},
        )
        return failure_result

    # ==================================================================================
    # Step 2: Convert FIRRTL to SystemVerilog using firtool
    # ==================================================================================
    context.logger.info("Step 2: Converting FIRRTL to SystemVerilog with firtool...")

    # Extract the simple class name from the full config name
    # e.g., "sims.palladium.BuckyballToyP2EConfig" -> "BuckyballToyP2EConfig"
    config_class_name = config_name.split(".")[-1]

    # Find the generated FIRRTL file (in base_build_dir, not verilog_output_dir)
    fir_file = f"{base_build_dir}/palladium.fpga.{config_class_name}.fir"
    if not os.path.exists(fir_file):
        context.logger.error(f"FIRRTL file not found: {fir_file}")
        context.logger.info(f"Looking for files in {base_build_dir}...")
        # List files to help debug
        if os.path.exists(base_build_dir):
            files = os.listdir(base_build_dir)
            context.logger.info(f"Files in build dir: {files}")
        success_result, failure_result = await check_result(
            context,
            1,
            continue_run=False,
            extra_fields={"task": "firrtl", "step": "file_check"},
        )
        return failure_result

    verilog_command = (
        f"firtool {fir_file} "
        f"-o {verilog_output_dir} "
        f"--split-verilog "
        f"--disable-all-randomization "
        f"--strip-debug-info "
        f"--ignore-read-enable-mem "
        f"--lowering-options=disallowLocalVariables "
        f"--disable-annotation-unknown"
    )

    result = stream_run_logger(
        cmd=verilog_command,
        logger=context.logger,
        cwd=arch_dir,
        stdout_prefix="palladium verilog",
        stderr_prefix="palladium verilog",
    )

    if result.returncode != 0:
        context.logger.error(f"Verilog generation failed with code {result.returncode}")
        success_result, failure_result = await check_result(
            context,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "verilog", "step": "firtool"},
        )
        return failure_result

    # ==================================================================================
    # Step 3: Verify and clean up
    # ==================================================================================
    # context.logger.info("Step 3: Verifying generated files...")

    # # Check if top-level Verilog was generated
    # top_sv_dir = f"{verilog_output_dir}/VCU118FPGATestHarness.sv"
    # top_sv_file = f"{top_sv_dir}/VCU118FPGATestHarness.sv"

    # if os.path.exists(top_sv_file):
    #     # Count generated files
    #     sv_files = [f for f in os.listdir(top_sv_dir) if f.endswith('.sv')]
    #     context.logger.info(f"Successfully generated {len(sv_files)} SystemVerilog files")
    #     context.logger.info(f"Top-level module: {top_sv_file}")
    # else:
    #     context.logger.error(f"Top-level Verilog file not found: {top_sv_file}")

    # # Remove unwanted file
    # topname_file = f"{arch_dir}/TestHarness.sv"
    # if os.path.exists(topname_file):
    #     os.remove(topname_file)
    #     context.logger.info(f"Removed unwanted file: {topname_file}")

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=data.get("from_run_workflow", False),
        extra_fields={
            "task": "verilog",
            "output_dir": verilog_output_dir,
            "top_module": "VCU118FPGATestHarness",
        },
    )

    # ==================================================================================
    # Continue routing
    # Routing to verilog or finish workflow
    # For run workflow, continue to verilog; for standalone clean, complete
    # ==================================================================================
    if data.get("from_run_workflow"):
        await context.emit(
            {"topic": "palladium.build", "data": {**data, "task": "run"}}
        )

    return
