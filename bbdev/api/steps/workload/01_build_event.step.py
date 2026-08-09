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

MODEL_TARGETS = {
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
    "buddynext": "buddy-buckyball-buddynext-all-run",
}

RUSHB_TARGETS = {
    "lenet": {
        "bemu": "buddy-buckyball-lenet-rushB-bemu-run",
        "verilator": "buddy-buckyball-lenet-rushB-verilator-run",
    },
    "mobilenet": {
        "bemu": "buddy-buckyball-mobilenetv3-rushB-bemu-run",
        "verilator": "buddy-buckyball-mobilenetv3-rushB-verilator-run",
    },
    "resnet": {
        "bemu": "buddy-buckyball-resnet-rushB-bemu-run",
        "verilator": "buddy-buckyball-resnet-rushB-verilator-run",
    },
    "yolo": {
        "bemu": "buddy-buckyball-yolo26-rushB-bemu-run",
        "verilator": "buddy-buckyball-yolo26-rushB-verilator-run",
    },
    "bert": {
        "bemu": "buddy-buckyball-bert-rushB-bemu-run",
        "verilator": "buddy-buckyball-bert-rushB-verilator-run",
    },
    "qwen3": {
        "bemu": "buddy-buckyball-qwen3-rushB-bemu-run",
        "verilator": "buddy-buckyball-qwen3-rushB-verilator-run",
    },
    "gemma4": {
        "bemu": "buddy-buckyball-gemma4-rushB-bemu-run",
        "verilator": "buddy-buckyball-gemma4-rushB-verilator-run",
    },
    "deepseekr1": {
        "bemu": "buddy-buckyball-deepseekr1-rushB-bemu-run",
        "verilator": "buddy-buckyball-deepseekr1-rushB-verilator-run",
    },
    "llama2": {
        "bemu": "buddy-buckyball-llama2-rushB-bemu-run",
        "verilator": "buddy-buckyball-llama2-rushB-verilator-run",
    },
    "stable-diffusion": {
        "bemu": "buddy-buckyball-stable-diffusion-rushB-bemu-run",
        "verilator": "buddy-buckyball-stable-diffusion-rushB-verilator-run",
    },
    "whisper": {
        "bemu": "buddy-buckyball-whisper-rushB-bemu-run",
        "verilator": "buddy-buckyball-whisper-rushB-verilator-run",
    },
    "buddynext": {
        "bemu": "buddy-buckyball-buddynext-rushB-bemu-run",
        "verilator": "buddy-buckyball-buddynext-rushB-verilator-run",
    },
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    workload_dir = f"{bbdir}/bb-tests"
    build_dir = f"{workload_dir}/build"
    allowed = {"chip", "model", "stable", "rushB", "_trace_id"}
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
    rushb_backend = input_data.get("rushB")

    if not isinstance(stable, bool):
        ctx.logger.error("Invalid parameter: stable must be a boolean flag")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_stable", "stable": stable},
            trace_id=origin_tid,
        )
        return

    if rushb_backend is not None and rushb_backend not in {"bemu", "verilator"}:
        ctx.logger.error("Invalid rushB backend: expected bemu or verilator")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_rushB", "rushB": rushb_backend},
            trace_id=origin_tid,
        )
        return

    target = ""
    if model:
        model_key = model.lower()
        target = MODEL_TARGETS.get(model_key)
        if target is None:
            ctx.logger.error(f"Unknown model: {model}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "unknown_model", "model": model},
                trace_id=origin_tid,
            )
            return
        if rushb_backend is not None:
            target = RUSHB_TARGETS.get(model_key, {}).get(rushb_backend)
            if target is None:
                ctx.logger.error(f"Missing rushB target registration for model {model}")
                await check_result(
                    ctx, 1, continue_run=False,
                    extra_fields={
                        "error": "missing_rushB_target",
                        "model": model,
                        "rushB": rushb_backend,
                    },
                    trace_id=origin_tid,
                )
                return
    elif rushb_backend is not None:
        target = f"rushB-{rushb_backend}-workloads-build"

    os.makedirs(build_dir, exist_ok=True)

    chip_arg = f"-DBUCKYBALL_WORKLOAD_CHIP={shlex.quote(chip)}"
    stable_arg = "-DBUCKYBALL_STABLE=ON" if stable else "-DBUCKYBALL_STABLE=OFF"
    ninja_target = f" {shlex.quote(target)}" if target else ""
    inner = (
        f"cd {shlex.quote(build_dir)} && "
        f"cmake -G Ninja {chip_arg} {stable_arg} "
        f"-DPython3_EXECUTABLE=\"$(which python3)\" .. && "
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
        ctx, result.returncode, continue_run=False,
        extra_fields={"chip": chip, "model": model, "rushB": rushb_backend},
        trace_id=origin_tid)

    # ==================================================================================
    #  finish workflow
    # ==================================================================================
    return
