import os
import shutil
import sys
import glob

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
    "name": "dc-verilog",
    "description": "generate RTL and memory metadata for downstream DC/tapeout flow",
    "flows": ["dc"],
    "triggers": [queue("dc.verilog")],
    "enqueues": [],
}


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
    elaborate_config = input_data.get("config")
    if not elaborate_config:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "validation", "error": "missing required parameter: config"},
            trace_id=origin_tid,
        )
        return failure_result
    build_dir = input_data.get("output_dir")
    if not build_dir:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "validation", "error": "missing required parameter: output_dir"},
            trace_id=origin_tid,
        )
        return failure_result
    ctx.logger.info(f"Using DC RTL output directory: {build_dir}")

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    mem_conf = os.path.join(build_dir, "dc.mems.conf")
    verilog_command = (
        f"mill -i __.test.runMain sims.verilator.Elaborate {elaborate_config} "
        "--disable-annotation-unknown --strip-debug-info -O=debug "
        "-lowering-options=disallowLocalVariables "
        f"--repl-seq-mem --repl-seq-mem-file={mem_conf} "
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
        continue_run=False,
        extra_fields={"task": "verilog", "source_list": source_list_path, "mem_conf": mem_conf},
        trace_id=origin_tid,
    )

    return
