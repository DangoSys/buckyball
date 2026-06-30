import os
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "workload-build",
    "description": "build workload",
    "flows": ["workload"],
    "triggers": [queue("workload.build")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    workload_dir = f"{bbdir}/bb-tests"
    build_dir = f"{workload_dir}/build"
    model = input_data.get("model", "")

    model_targets = {
        "lenet": "buddy-buckyball-lenet-run",
        "mobilenet": "buddy-buckyball-mobilenetv3-run",
        "resnet": "buddy-buckyball-resnet-run",
        "yolo": "buddy-buckyball-yolo26-run",
        "bert": "buddy-buckyball-bert-run",
        "qwen3": "buddy-buckyball-qwen3-run",
        "gemma4": "buddy-buckyball-gemma4-run",
        "deepseekr1": "buddy-buckyball-deepseekr1-run",
        "llama2": "buddy-buckyball-llama2-run",
        "stable-diffusion": "buddy-buckyball-stable-diffusion-run",
        "whisper": "buddy-buckyball-whisper-run",
    }
    target = ""
    if model:
        model_key = model.lower()
        target = model_targets.get(model_key)
        if target is None:
            ctx.logger.error(f"Unknown model: {model}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "unknown_model", "model": model},
                trace_id=origin_tid,
            )
            return

    os.makedirs(build_dir, exist_ok=True)

    ninja_target = f" {target}" if target else ""
    command = f"cd {bbdir} && nix develop -c bash -c 'cd {build_dir} && cmake -G Ninja .. && ninja -j{os.cpu_count()}{ninja_target}'"
    ctx.logger.info(
        "Executing workload command", {"command": command, "cwd": build_dir}
    )
    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=workload_dir,
        executable="bash",
        stdout_prefix="workload build",
        stderr_prefix="workload build",
    )

    # ==================================================================================
    # Return simulation result
    # ==================================================================================
    # This is the end of run workflow, status no longer set to processing
    success_result, failure_result = await check_result(
        ctx, result.returncode, continue_run=False, trace_id=origin_tid)

    # ==================================================================================
    #  finish workflow
    # ==================================================================================
    return
