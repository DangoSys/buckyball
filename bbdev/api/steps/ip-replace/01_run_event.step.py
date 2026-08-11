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

from sram_replace import prepare_ip_replacement


config = {
    "name": "ip-replace",
    "description": "replace behavioral SRAM/IP RTL with compiler macro wrappers",
    "flows": ["ip-replace", "dc", "yosys"],
    "triggers": [queue("ip-replace.run")],
    "enqueues": ["yosys.synth"],
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
    output_dir = input_data.get("ip_replace_output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        output_dir = os.path.join(os.path.dirname(source_list_path), "ip-replace")
    os.makedirs(output_dir, exist_ok=True)

    try:
        replacement = prepare_ip_replacement(
            sources,
            output_dir,
            os.environ.get("SKY130_ROOT"),
            input_data.get("top"),
        )
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
        for path in replacement["source_paths"]:
            handle.write(f"{path}\n")

    replacement["source_list"] = replaced_source_list
    extra = {
        "task": "ip-replace",
        "consumer": consumer,
        "source_list": replaced_source_list,
        "ip_manifest": replacement.get("manifest"),
        "ip_mapped_memories": replacement["mapped"],
        "ip_macro_area": replacement["macro_area"],
    }
    if input_data.get("mem_conf"):
        extra["mem_conf"] = input_data["mem_conf"]
    if replacement["mapped"]:
        ctx.logger.info(
            f"Mapped {replacement['mapped']} behavioral memories to compiler macros; "
            f"estimated macro area {replacement['macro_area']:.3f} um^2"
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
                    "ip_replacement": replacement,
                    "_trace_id": origin_tid,
                },
            }
        )
        return

    await check_result(ctx, 0, continue_run=False, extra_fields=extra, trace_id=origin_tid)
