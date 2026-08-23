import os
import sys
import glob
import re

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
step_path = os.path.dirname(__file__)
if step_path not in sys.path:
    sys.path.insert(0, step_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.chip import require_chip
from utils.path import get_buckyball_path, get_dc_analysis_dir
from utils.rtl import run_chip_mill

config = {
    "name": "dc-verilog",
    "description": "generate RTL and memory metadata for downstream DC/tapeout flow",
    "flows": ["dc"],
    "triggers": [queue("dc.verilog")],
    "enqueues": ["ip-replace.run"],
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
    try:
        chip = require_chip(input_data)
        build_dir, returncode = await run_chip_mill(
            ctx, bbdir, chip, "synth", "dc verilog",
            input_data.get("output_dir"),
        )
    except (ValueError, RuntimeError) as error:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "validation", "error": str(error)},
            trace_id=origin_tid,
        )
        return failure_result
    if returncode != 0:
        _, failure_result = await check_result(
            ctx,
            returncode,
            continue_run=False,
            extra_fields={"task": "verilog"},
            trace_id=origin_tid,
        )
        return failure_result
    top_module = input_data.get("top") or "DigitalTop"
    ctx.logger.info(f"Using DC RTL output directory: {build_dir}, synthesis top: {top_module}")
    mem_conf = os.path.join(build_dir, "mems.conf")

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
        0,
        continue_run=True,
        extra_fields={
            "task": "verilog",
            "source_list": source_list_path,
            "mem_conf": mem_conf,
            "top_module": top_module,
        },
        trace_id=origin_tid,
    )

    payload = {
        **input_data,
        "source_list": source_list_path,
        "ip_replace_output_dir": os.path.join(build_dir, "ip-replace"),
        "consumer": "dc",
        "mem_conf": mem_conf,
        "top": top_module,
        "next_topic": "dc.area" if input_data.get("from_area_workflow") or input_data.get("from_power_workflow") else None,
        "_trace_id": origin_tid,
    }
    if input_data.get("from_power_workflow"):
        payload["analysis_dir"] = get_dc_analysis_dir(bbdir, chip, "power")
    elif input_data.get("from_area_workflow"):
        payload["analysis_dir"] = get_dc_analysis_dir(bbdir, chip, "area")
    await ctx.enqueue({"topic": "ip-replace.run", "data": payload})

    return
