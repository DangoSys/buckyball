import os
import sys
from pathlib import Path

import yaml

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import check_result, get_origin_trace_id, require_design
from utils.path import get_buckyball_path, rtl_dir
from steps.dc.scripts.tapeout import get_tapeout_contract
from generate import generate_sram

config = {
    "name": "ip-generate",
    "description": "generate SRAM macro wrappers from elaborator mems.conf via the process sram function",
    "flows": ["ip", "dc", "yosys"],
    "triggers": [queue("ip.generate")],
    "enqueues": ["ip.replace"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        design = require_design(input_data)
        consumer = input_data.get("consumer") or "dc"
        if consumer not in ("dc", "yosys"):
            raise ValueError("consumer must be dc or yosys")
        # The synthesis top: --top, else the design's tapeout config (dc) or the yosys config.
        top = input_data.get("top") or (
            get_tapeout_contract(
                get_buckyball_path(), design, None, selection=input_data["target"]
            ).top_module
            if consumer == "dc"
            else yaml.safe_load(
                (
                    Path(__file__).parents[1] / "yosys/scripts/yosys-config.yaml"
                ).read_text()
            )["top"]
        )
        if not isinstance(top, str) or not top:
            raise ValueError("top must be a non-empty string")
        bbdir = Path(get_buckyball_path()).resolve()
        build_dir = Path(
            rtl_dir(
                bbdir,
                design,
                "dc" if consumer == "dc" else "synth",
                input_data.get("output_dir"),
                selection=input_data["target"],
            )
        ).resolve()
        man = generate_sram(
            bbdir=bbdir,
            design=design,
            selection=input_data["target"],
            build_dir=build_dir,
            top_module=top,
            source_list=build_dir / f"{consumer}_sources.list",
        )
    except Exception as exc:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "ip.generate", "error": str(exc)},
            trace_id=origin_tid,
        )
        return failure_result

    extra = {
        "task": "ip.generate",
        "design": man["design"],
        "generate_manifest": man["generate_manifest"],
        "sram_macros_v": man["sram_macros_v"],
        "mem_count": man["mem_count"],
        "build_dir": str(build_dir),
    }
    await check_result(
        ctx, 0, continue_run=True, extra_fields=extra, trace_id=origin_tid
    )
    await ctx.enqueue(
        {
            "topic": "ip.replace",
            "data": {
                **input_data,
                "design": design,
                "consumer": consumer,
                "top": top,
                "_trace_id": origin_tid,
                "target": input_data["target"],
            },
        }
    )
