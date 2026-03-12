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
    "subscribes": ["verilator.verilog"],
    "emits": ["verilator.build"],
    "flows": ["verilator"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    build_dir = data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"
    config_name = data.get("config")
    
    if not config_name or config_name == "None":
        context.logger.error("Configuration name is required but not provided")
        success_result, failure_result = await check_result(
            context,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
        )
        return failure_result

    context.logger.info(f"Using configuration: {config_name}")

    # ==================================================================================
    # Execute operation
    # ==================================================================================
    if data.get("balltype"):
        command = (
            f"mill -i __.test.runMain sims.verify.BallTopMain {data.get('balltype')} "
        )
    else:
        command = f"mill -i __.test.runMain sims.verilator.BBSimElaborate {config_name} "

    command += "--disable-annotation-unknown -strip-debug-info -O=debug "
    command += f"--split-verilog -o={build_dir}"

    result = stream_run_logger(
        cmd=command,
        logger=context.logger,
        cwd=arch_dir,
        stdout_prefix="verilator verilog",
        stderr_prefix="verilator verilog",
    )

    # Remove testchipip C++ sources that depend on fesvr (which we don't have).
    # SimTSI.v is kept so verilator can resolve the SimTSI module reference in BBSimHarness.sv;
    # tsi_tick DPI symbol is satisfied by arch/src/csrc/src/monitor/ioe/tsi_stub.cc instead.
    for unwanted in [
        f"{build_dir}/testchip_htif.cc",
        f"{build_dir}/testchip_htif.h",
        f"{build_dir}/testchip_tsi.cc",
        f"{build_dir}/testchip_tsi.h",
        f"{build_dir}/SimTSI.cc",
        f"{arch_dir}/BBSimHarness.sv",
    ]:
        if os.path.exists(unwanted):
            os.remove(unwanted)

    # Patch fesvr includes out of mm.h and mm.cc (copied from testchipip resources).
    # They reference fesvr/memif.h which we don't have — our SimDRAM_bb.cc doesn't use it.
    for patch_file in [f"{build_dir}/mm.h", f"{build_dir}/mm.cc"]:
        if os.path.exists(patch_file):
            with open(patch_file, "r") as f:
                content = f.read()
            patched = "\n".join(
                line for line in content.splitlines()
                if "fesvr/memif.h" not in line and "fesvr/elfloader.h" not in line
            )
            if patched != content:
                with open(patch_file, "w") as f:
                    f.write(patched)
                context.logger.info(f"Patched fesvr includes from {patch_file}")

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        context,
        result.returncode,
        continue_run=data.get("from_run_workflow", False),
        extra_fields={"task": "verilog"},
    )

    # ==================================================================================
    # Continue routing
    # Routing to verilog or finish workflow
    # For run workflow, continue to verilog; for standalone clean, complete
    # ==================================================================================
    if data.get("from_run_workflow"):
        await context.emit(
            {"topic": "verilator.build", "data": {**data, "task": "run"}}
        )

    return
