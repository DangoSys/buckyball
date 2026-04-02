import os
import subprocess
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-verilog",
    "description": "generate verilog code",
    "flows": ["verilator"],
    "triggers": [queue("verilator.verilog")],
    "enqueues": ["verilator.build"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"
    config_name = input_data.get("config")

    if not config_name or config_name == "None":
        ctx.logger.error("Configuration name is required but not provided")
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "validation",
                "error": "Configuration name is required. Please specify --config parameter.",
                "example": 'bbdev verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
            },
            trace_id=origin_tid,
        )
        return failure_result

    ctx.logger.info(f"Using configuration: {config_name}")

    # ==================================================================================
    # Execute operation
    # ==================================================================================
    if input_data.get("balltype"):
        command = (
            f"mill -i __.test.runMain sims.verify.BallTopMain {input_data.get('balltype')} "
        )
    else:
        command = f"mill -i __.test.runMain sims.verilator.Elaborate {config_name} "

    # Firtool options (CIRCT). Current set; optional Chipyard-style options below.
    command += "--disable-annotation-unknown "
    command += "--strip-debug-info "
    command += "-O=debug "
    # command += f"-repl-seq-mem -repl-seq-mem-file={build_dir}/mem.conf "
    command += f"--split-verilog -o={build_dir} "
    # Optional: --disable-annotation-classless (ignore classless annotations)
    # Optional: -repl-seq-mem -repl-seq-mem-file=<path>.conf (SRAM macro replacement)
    # Optional: --disable-all-randomization (disable mem/reg init; may break semantics)
    # Optional: --disable-opt (no optimization) or -O=release (default is release)
    # Optional: --output-annotation-file=<path> (emit annotations after lower-to-hw)
    # Optional: --no-dedup (disable module dedup); --strip-fir-debug-info (FIR locators)

    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
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
                ctx.logger.info(f"Patched fesvr includes from {patch_file}")

    # ==================================================================================
    # Return result to API
    # ==================================================================================
    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog"},
        trace_id=origin_tid,
    )

    # ==================================================================================
    # Continue routing
    # ==================================================================================
    if input_data.get("from_run_workflow"):
        await ctx.enqueue(
            {"topic": "verilator.build", "data": {**input_data, "task": "run"}}
        )

    return
