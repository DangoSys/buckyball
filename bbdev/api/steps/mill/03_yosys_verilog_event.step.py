import os
import shutil
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
    "name": "yosys verilog",
    "description": "generate verilog for yosys flow",
    "flows": ["yosys"],
    "triggers": [queue("yosys.run"), queue("yosys.verilog")],
    "enqueues": ["yosys.synth"],
}


def load_yosys_config():
    bbdir = get_buckyball_path()
    config_path = f"{bbdir}/bbdev/api/steps/yosys/scripts/yosys-config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"

    yosys_cfg = load_yosys_config()
    elaborate_config = input_data.get("config") or yosys_cfg.get(
        "elaborate_config", "sims.verilator.BuckyballToyVerilatorConfig"
    )

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    verilog_command = (
        f"mill -i __.test.runMain sims.verilator.Elaborate {elaborate_config} "
        "--disable-annotation-unknown -strip-debug-info -O=debug "
        "-lowering-options=disallowLocalVariables "
        f"--split-verilog -o={build_dir}"
    )

    result = stream_run_logger(
        cmd=verilog_command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="yosys verilog",
        stderr_prefix="yosys verilog",
    )

    if result.returncode != 0:
        success_result, failure_result = await check_result(
            ctx,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "verilog"},
            trace_id=origin_tid,
        )
        return failure_result

    for unwanted in ["TestHarness.sv", "TargetBall.sv"]:
        topname_file = f"{arch_dir}/{unwanted}"
        if os.path.exists(topname_file):
            os.remove(topname_file)

    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog"},
        trace_id=origin_tid,
    )

    if input_data.get("from_run_workflow"):
        await ctx.enqueue({"topic": "yosys.synth", "data": {**input_data, "task": "run"}})

    return
