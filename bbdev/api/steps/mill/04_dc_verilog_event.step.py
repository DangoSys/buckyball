import os
import shutil
import sys
import glob
import yaml

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
    "name": "dc-verilog",
    "description": "generate verilog for dc flow",
    "flows": ["dc"],
    "triggers": [queue("dc.run"), queue("dc.verilog")],
    "enqueues": ["dc.synth"],
}


def load_dc_config():
    bbdir = get_buckyball_path()
    config_path = f"{bbdir}/bbdev/api/steps/dc/scripts/dc-config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def prepare_dc_verilog(build_dir: str):
    vsrcs = sorted(
        glob.glob(f"{build_dir}/**/*.sv", recursive=True)
        + glob.glob(f"{build_dir}/**/*.v", recursive=True)
    )
    if not vsrcs:
        raise RuntimeError("no dc verilog source generated")
    source_list_path = os.path.join(build_dir, "dc_sources.list")
    with open(source_list_path, "w") as f:
        for path in vsrcs:
            f.write(path + "\n")
    return source_list_path


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = input_data.get("output_dir") or f"{bbdir}/arch/build/"

    dc_cfg = load_dc_config()
    elaborate_config = (
        input_data.get("config")
        or dc_cfg.get("elaborate_config")
        or "sims.verilator.BuckyballToyVerilatorConfig"
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
        stdout_prefix="dc verilog",
        stderr_prefix="dc verilog",
    )

    if result.returncode != 0:
        _, failure_result = await check_result(
            ctx,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "verilog"},
            trace_id=origin_tid,
        )
        return failure_result

    try:
        source_list_path = prepare_dc_verilog(build_dir)
    except Exception as e:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "verilog", "error": str(e)},
            trace_id=origin_tid,
        )
        return failure_result

    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog", "source_list": source_list_path},
        trace_id=origin_tid,
    )

    if input_data.get("from_run_workflow"):
        await ctx.enqueue({"topic": "dc.synth", "data": {**input_data, "task": "run"}})

    return
