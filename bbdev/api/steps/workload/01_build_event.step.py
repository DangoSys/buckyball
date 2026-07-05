import os
import sys
import shlex
import re
from pathlib import Path

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
    allowed = {"chip", "model", "stable", "_trace_id"}
    unknown = sorted(k for k in input_data if k not in allowed)
    if unknown:
        ctx.logger.error(f"Unknown workload build parameter(s): {', '.join(unknown)}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "unknown_parameter", "parameters": unknown},
            trace_id=origin_tid,
        )
        return
    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip must be specified")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    if not isinstance(chip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        ctx.logger.error(f"Invalid chip: {chip}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return
    chip_dir = Path(bbdir) / "examples" / "chips" / chip
    if not chip_dir.is_dir():
        ctx.logger.error(f"Workload chip does not exist: {chip}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "unknown_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return
    model = input_data.get("model", "")
    stable = input_data.get("stable", False)

    if not isinstance(stable, bool):
        ctx.logger.error("Invalid parameter: stable must be a boolean flag")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_stable", "stable": stable},
            trace_id=origin_tid,
        )
        return

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

    chip_arg = f"-DBUCKYBALL_WORKLOAD_CHIP={shlex.quote(chip)}"
    stable_arg = "-DBUCKYBALL_STABLE=ON" if stable else "-DBUCKYBALL_STABLE=OFF"
    ninja_target = f" {shlex.quote(target)}" if target else ""
    inner = (
        f"cd {shlex.quote(build_dir)} && "
        f"cmake -G Ninja {chip_arg} {stable_arg} .. && "
        f"ninja -j{os.cpu_count()}{ninja_target}"
    )
    command = f"cd {shlex.quote(bbdir)} && nix develop -c bash -c {shlex.quote(inner)}"
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
