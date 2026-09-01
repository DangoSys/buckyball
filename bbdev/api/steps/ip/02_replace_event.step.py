import os
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.event_common import check_result, get_origin_trace_id, require_chip
from utils.path import get_buckyball_path, rtl_dir
from replace import replace_sources

_SOURCE_LIST = {"dc": "dc_sources.list", "yosys": "yosys_sources.list"}


config = {
    "name": "ip-replace",
    "description": "assemble top-scoped synthesis RTL plus generated SRAM macros",
    "flows": ["ip", "dc", "yosys"],
    "triggers": [queue("ip.replace")],
    "enqueues": ["dc.area", "yosys.synth"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
        consumer = input_data.get("consumer") or "dc"
        if consumer not in _SOURCE_LIST:
            raise ValueError("consumer must be dc or yosys")
        top_module = input_data.get("top") or "DigitalTop"
        if not isinstance(top_module, str) or not top_module:
            raise ValueError("top must be a non-empty string")

        bbdir = Path(get_buckyball_path()).resolve()
        build_dir = Path(rtl_dir(bbdir, chip, "verilog", input_data.get("output_dir"))).resolve()
        source_list_path = build_dir / _SOURCE_LIST[consumer]
        if not source_list_path.is_file():
            raise FileNotFoundError(f"missing source list: {source_list_path}")
        sources = [line.strip() for line in source_list_path.read_text().splitlines() if line.strip()]
        if not sources:
            raise ValueError(f"empty source list: {source_list_path}")

        gen_dir = build_dir / "ip-generate"
        sram_macros_v = gen_dir / "sram_macros.v"
        generate_manifest = gen_dir / "generate_manifest.json"
        if not generate_manifest.is_file():
            raise FileNotFoundError(f"missing generate_manifest: {generate_manifest}")

        result = replace_sources(
            source_paths=sources,
            top_module=top_module,
            sram_macros_v=sram_macros_v,
            output_dir=build_dir / "ip-replace",
            consumer=consumer,
            generate_manifest=generate_manifest,
        )
    except Exception as exc:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "ip.replace", "error": str(exc)},
            trace_id=origin_tid,
        )
        return failure_result

    extra = {
        "task": "ip.replace",
        "consumer": consumer,
        "source_list": result["source_list"],
        "replace_manifest": result["replace_manifest"],
        "source_count": result["source_count"],
        "top_module": top_module,
        "build_dir": str(build_dir),
    }
    if generate_manifest.is_file():
        extra["generate_manifest"] = str(generate_manifest)

    next_topic = input_data.get("next_topic")
    if next_topic is not None:
        if not isinstance(next_topic, str) or not next_topic:
            _, failure_result = await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={"task": "ip.replace", "error": "next_topic must be a non-empty string"},
                trace_id=origin_tid,
            )
            return failure_result
        await check_result(ctx, 0, continue_run=True, extra_fields=extra, trace_id=origin_tid)
        await ctx.enqueue(
            {
                "topic": next_topic,
                "data": {
                    **input_data,
                    "chip": chip,
                    "consumer": consumer,
                    "top": top_module,
                    "source_list": result["source_list"],
                    "replace_manifest": result["replace_manifest"],
                    "generate_manifest": str(generate_manifest),
                    "_trace_id": origin_tid,
                },
            }
        )
        return

    await check_result(ctx, 0, continue_run=False, extra_fields=extra, trace_id=origin_tid)
