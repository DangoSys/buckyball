import os
import sys

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result


config = {
    "type": "event",
    "name": "make pegasus verilog",
    "description": "Generate SystemVerilog from Chisel using ElaboratePegasus",
    "subscribes": ["pegasus.verilog"],
    "emits": [],
    "flows": ["pegasus"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = data.get("output_dir", f"{bbdir}/arch/build/pegasus/")

    config_name = data.get("config", "sims.pegasus.PegasusConfig")
    context.logger.info(f"[pegasus] Elaborating config: {config_name}")
    context.logger.info(f"[pegasus] Output directory: {build_dir}")

    os.makedirs(build_dir, exist_ok=True)

    # Use ElaboratePegasus (not the generic sims.verilator.Elaborate)
    # ElaboratePegasus instantiates PegasusHarness directly, bypassing TestHarness
    command = (
        f"mill -i __.test.runMain sims.pegasus.ElaboratePegasus "
        f"--disable-annotation-unknown "
        f"-strip-debug-info "
        f"-O=debug "
        f"--split-verilog "
        f"-o={build_dir}"
    )

    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=arch_dir,
        stdout_prefix="pegasus verilog",
        stderr_prefix="pegasus verilog",
    )

    # Clean up stray top-level file if emitted next to arch/
    for stray in ["PegasusHarness.sv", "TestHarness.sv"]:
        stray_path = f"{arch_dir}/{stray}"
        if os.path.exists(stray_path):
            os.remove(stray_path)

    # Copy generated SV files to pegasus/vivado/generated/ for Vivado build
    vivado_gen_dir = f"{bbdir}/pegasus/vivado/generated"
    if result.returncode == 0 and os.path.isdir(build_dir):
        import shutil
        os.makedirs(vivado_gen_dir, exist_ok=True)
        sv_files = [f for f in os.listdir(build_dir) if f.endswith(".sv") or f.endswith(".v")]
        for f in sv_files:
            shutil.copy2(os.path.join(build_dir, f), os.path.join(vivado_gen_dir, f))
        context.logger.info(f"[pegasus] Copied {len(sv_files)} files to {vivado_gen_dir}")

    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilog",
            "output_dir": build_dir,
            "vivado_gen_dir": vivado_gen_dir,
            "top_module": "PegasusHarness",
        },
    )

    return
