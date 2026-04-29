import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.path import get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "make p2e verilog",
    "description": "Generate DigitalTop RTL and P2E DDR4 backdoor wrapper",
    "flows": ["p2e"],
    "triggers": [queue("p2e.verilog")],
    "enqueues": [],
}


def cleanup_strays(arch_dir: str):
    for stray in ["P2EHarness.sv", "P2ETop.v", "P2ETopWrapper.sv"]:
        path = os.path.join(arch_dir, stray)
        if os.path.exists(path):
            os.remove(path)


def normalize_p2e_timescale(build_dir: str, logger):
    patched = 0
    for root, _, files in os.walk(build_dir):
        for name in files:
            if not name.endswith((".v", ".sv")):
                continue

            path = os.path.join(root, name)
            with open(path, "r") as f:
                content = f.read()

            if "`timescale" in content:
                continue

            with open(path, "w") as f:
                f.write("`timescale 1ns/1ps\n")
                f.write(content)
            patched += 1

    logger.info(f"Normalized P2E timescale in {patched} generated Verilog files")


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    config_name = input_data.get("config", "sims.p2e.P2EToyConfig")
    build_dir = get_verilator_build_dir(bbdir, config_name, input_data.get("output_dir"))

    ctx.logger.info(f"Using P2E configuration: {config_name}")
    ctx.logger.info(f"Using P2E output directory: {build_dir}")

    os.makedirs(build_dir, exist_ok=True)
    cleanup_strays(arch_dir)

    common_firtool_opts = (
        "--disable-annotation-unknown "
        "--strip-debug-info "
        "-O=debug "
        f"--split-verilog -o={build_dir}"
    )

    digital_command = (
        f"mill -i __.test.runMain sims.p2e.Elaborate {config_name} "
        f"{common_firtool_opts}"
    )
    digital_result = stream_run_logger(
        cmd=digital_command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="p2e digital verilog",
        stderr_prefix="p2e digital verilog",
    )
    if digital_result.returncode != 0:
        cleanup_strays(arch_dir)
        await check_result(
            ctx,
            digital_result.returncode,
            continue_run=False,
            extra_fields={"task": "verilog", "step": "digital", "config": config_name},
            trace_id=origin_tid,
        )
        return

    top_command = (
        f"mill -i __.test.runMain sims.p2e.ElaborateP2ETop "
        f"{common_firtool_opts}"
    )
    top_result = stream_run_logger(
        cmd=top_command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="p2e top verilog",
        stderr_prefix="p2e top verilog",
    )

    cleanup_strays(arch_dir)
    if top_result.returncode == 0:
        normalize_p2e_timescale(build_dir, ctx.logger)

    await check_result(
        ctx,
        top_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilog",
            "config": config_name,
            "output_dir": build_dir,
            "top_module": "P2ETop",
        },
        trace_id=origin_tid,
    )
