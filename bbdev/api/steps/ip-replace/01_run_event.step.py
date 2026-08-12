import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from sram_replace import prepare_sram_collateral


config = {
    "name": "ip-replace",
    "description": "prepare top-scoped synthesis RTL and SRAM metadata",
    "flows": ["ip-replace", "dc", "yosys"],
    "triggers": [queue("ip-replace.run")],
    "enqueues": ["dc.area", "yosys.synth"],
}


def default_source_list(input_data: dict) -> str | None:
    source_list = input_data.get("source_list")
    if isinstance(source_list, str) and source_list:
        return source_list
    build_dir = input_data.get("output_dir")
    consumer = input_data.get("consumer")
    if not isinstance(build_dir, str):
        return None
    if consumer == "dc":
        return os.path.join(build_dir, "dc_sources.list")
    if consumer == "yosys":
        return os.path.join(build_dir, "yosys_sources.list")
    return None


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    source_list_path = default_source_list(input_data)
    if not source_list_path or not os.path.isfile(source_list_path):
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "ip-replace", "error": "missing source list"},
            trace_id=origin_tid,
        )
        return failure_result

    with open(source_list_path) as handle:
        sources = [line.strip() for line in handle if line.strip()]
    if not sources:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "ip-replace", "error": "empty source list"},
            trace_id=origin_tid,
        )
        return failure_result

    consumer = input_data.get("consumer", "generic")
    top_module = input_data.get("top") or "DigitalTop"
    output_dir = input_data.get("ip_replace_output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        output_dir = os.path.join(os.path.dirname(source_list_path), "ip-replace")
    os.makedirs(output_dir, exist_ok=True)

    try:
        collateral = prepare_sram_collateral(sources, output_dir, top_module)
    except Exception as exc:
        _, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "ip-replace", "error": str(exc)},
            trace_id=origin_tid,
        )
        return failure_result

    replaced_source_list = os.path.join(output_dir, f"{consumer}_sources.list")
    with open(replaced_source_list, "w") as handle:
        for path in collateral["source_paths"]:
            handle.write(f"{path}\n")

    collateral["source_list"] = replaced_source_list
    extra = {
        "task": "ip-replace",
        "consumer": consumer,
        "source_list": replaced_source_list,
        "sram_manifest": collateral["sram_manifest"],
        "sram_memory_count": collateral["sram_memory_count"],
        "top_module": collateral["top_module"],
    }
    if input_data.get("mem_conf"):
        extra["mem_conf"] = input_data["mem_conf"]
    ctx.logger.info(
        f"Prepared {collateral['sram_memory_count']} technology-neutral SRAM modules "
        f"for top {collateral['top_module']}"
    )

    next_topic = input_data.get("next_topic")
    if isinstance(next_topic, str) and next_topic:
        await check_result(ctx, 0, continue_run=True, extra_fields=extra, trace_id=origin_tid)
        await ctx.enqueue(
            {
                "topic": next_topic,
                "data": {
                    **input_data,
                    "source_list": replaced_source_list,
                    "sram_collateral": collateral,
                    "_trace_id": origin_tid,
                },
            }
        )
        return

    await check_result(ctx, 0, continue_run=False, extra_fields=extra, trace_id=origin_tid)
