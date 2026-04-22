import os
import shutil
import sys
import yaml

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
    "name": "dc-synth",
    "description": "run Design Compiler synthesis",
    "flows": ["dc"],
    "triggers": [queue("dc.synth")],
    "enqueues": [],
}


def load_dc_config():
    config_path = os.path.join(os.path.dirname(__file__), "scripts", "dc-config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def prepare_dc_design(build_dir: str, design_dir: str):
    source_list_path = os.path.join(build_dir, "dc_sources.list")
    if not os.path.exists(source_list_path):
        raise RuntimeError("missing dc_sources.list, run dc verilog first")
    with open(source_list_path, "r") as f:
        sources = [line.strip() for line in f.readlines() if line.strip()]
    if not sources:
        raise RuntimeError("empty dc_sources.list")
    for src in sources:
        rel = os.path.relpath(src, build_dir)
        dst = os.path.join(design_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return source_list_path


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = input_data.get("output_dir") or f"{bbdir}/arch/build/"

    dc_cfg = load_dc_config()
    top_module = input_data.get("top") or dc_cfg.get("top") or "BuckyballAccelerator"

    work_dir = f"{bbdir}/bb-tests/output/dc"
    design_dir = f"{work_dir}/design"
    report_dir = f"{work_dir}/reports"
    tmp_dir = f"{work_dir}/tmp"

    for d in (design_dir, report_dir, tmp_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    try:
        source_list_path = prepare_dc_design(build_dir, design_dir)
    except Exception as e:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "dc", "error": str(e)},
            trace_id=origin_tid,
        )
        return

    tcl_script = os.path.join(os.path.dirname(__file__), "scripts", "run-dc.tcl")
    target_libs = " ".join(dc_cfg.get("target_libraries") or [])
    setup = (
        f'set design_dir "{design_dir}"; '
        f'set top_module "{top_module}"; '
        f'set work_dir "{work_dir}"; '
        f'set report_dir "{report_dir}"; '
        f'set tmp_dir "{tmp_dir}"; '
        f'set target_libs "{target_libs}"; '
        f'set clock_name "{dc_cfg.get("clock_name", "clock")}"; '
        f'set clock_period {dc_cfg.get("clock_period", 2.0)}; '
        f'set clock_uncertainty {dc_cfg.get("clock_uncertainty", 0.6)}; '
        f'set clock_transition {dc_cfg.get("clock_transition", 0.08)}; '
        f'set input_delay {dc_cfg.get("input_delay", 1.2)}; '
        f'set output_delay {dc_cfg.get("output_delay", 0.6)}; '
        f'set input_transition {dc_cfg.get("input_transition", 0.2)}; '
        f'set output_load {dc_cfg.get("output_load", 0.08)};'
    )
    command = f'dc_shell -x \'{setup}\' -f {tcl_script}'

    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="dc run",
        stderr_prefix="dc run",
    )

    alib = os.path.join(bbdir, "alib-52")
    if os.path.exists(alib):
        shutil.rmtree(alib)

    extra = {"task": "dc", "report_dir": report_dir, "top": top_module, "source_list": source_list_path}
    for name in ("area.rpt", "hierarchy.rpt", "timing.rpt", "power.rpt"):
        path = os.path.join(report_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                extra[name] = f.read()

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=extra,
        trace_id=origin_tid,
    )

    return
