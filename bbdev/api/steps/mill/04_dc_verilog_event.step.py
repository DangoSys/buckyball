import os
import shutil
import sys
import glob
import re

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


def is_dpi_source(path: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return 'import "DPI-C"' in f.read()


def build_stub_from_header(src_path: str) -> str:
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    m = re.search(
        r"module\s+([A-Za-z_][A-Za-z0-9_]*)\s*(#\s*\(.*?\)\s*)?\((.*?)\)\s*;",
        content,
        re.S,
    )
    if m is None:
        raise RuntimeError(f"invalid module header format: {src_path}")
    mod_name = m.group(1)
    params_block = m.group(2) or ""
    ports_block = m.group(3).rstrip()
    return f"(* blackbox *) module {mod_name} {params_block}(\n{ports_block}\n);\nendmodule\n"


def prepare_dc_verilog(build_dir: str):
    vsrcs = sorted(
        glob.glob(f"{build_dir}/**/*.sv", recursive=True)
        + glob.glob(f"{build_dir}/**/*.v", recursive=True)
    )
    stub_dir = os.path.join(build_dir, "dc_stubs")
    os.makedirs(stub_dir, exist_ok=True)
    kept = []
    stubbed_dpi = []
    for path in vsrcs:
        if is_dpi_source(path):
            stub_path = os.path.join(stub_dir, f"stub_{os.path.basename(path)}")
            with open(stub_path, "w") as f:
                f.write(build_stub_from_header(path))
            kept.append(stub_path)
            stubbed_dpi.append(path)
        else:
            kept.append(path)
    if not kept:
        raise RuntimeError("no dc verilog source generated")
    source_list_path = os.path.join(build_dir, "dc_sources.list")
    with open(source_list_path, "w") as f:
        for path in kept:
            f.write(path + "\n")
    return source_list_path, stubbed_dpi


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
        source_list_path, stubbed_dpi = prepare_dc_verilog(build_dir)
    except Exception as e:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "verilog", "error": str(e)},
            trace_id=origin_tid,
        )
        return failure_result
    if stubbed_dpi:
        ctx.logger.info(
            f"Stubbed {len(stubbed_dpi)} DPI-C sources for DC: "
            f"{', '.join(os.path.basename(path) for path in stubbed_dpi[:10])}"
            f"{'...' if len(stubbed_dpi) > 10 else ''}"
        )

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"task": "verilog", "source_list": source_list_path, "mem_conf": mem_conf},
        trace_id=origin_tid,
    )

    return
