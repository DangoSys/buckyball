import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "generate pegasus bitstream",
    "description": "generate pegasus bitstream",
    "flows": ["pegasus"],
    "triggers": [queue("pegasus.buildbitstream")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    generated_dir = input_data.get("generated_dir", f"{bbdir}/thirdparty/pegasus/vivado/generated")
    output_dir = input_data.get("output_dir", f"{bbdir}/thirdparty/pegasus/vivado/build")
    top_module = input_data.get("top", "PegasusTop")

    ctx.logger.info(f"[pegasus] Generated dir: {generated_dir}")
    ctx.logger.info(f"[pegasus] Output dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(generated_dir):
        ctx.logger.error(f"[pegasus] generated dir not found: {generated_dir}")
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "buildbitstream", "error": "missing generated_dir"},
            trace_id=origin_tid,
        )
        return failure_result

    has_rtl = any(name.endswith(".sv") or name.endswith(".v") for name in os.listdir(generated_dir))
    if not has_rtl:
        ctx.logger.error(f"[pegasus] no verilog files found in: {generated_dir}")
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "buildbitstream", "error": "empty generated_dir"},
            trace_id=origin_tid,
        )
        return failure_result

    bit_cmd = (
        f"bash {bbdir}/thirdparty/pegasus/vivado/build-bitstream.sh "
        f"--source_dir {generated_dir} "
        f"--output_dir {output_dir} "
        f"--top {top_module}"
    )
    result = stream_run_logger(
        cmd=bit_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="pegasus bitstream",
        stderr_prefix="pegasus bitstream",
    )

    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "buildbitstream",
            "output_dir": output_dir,
            "bitstream": f"{output_dir}/{top_module}.bit",
        },
        trace_id=origin_tid,
    )
    return
