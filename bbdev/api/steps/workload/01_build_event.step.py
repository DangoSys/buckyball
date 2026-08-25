import importlib.util
import os
import sys
import re
from pathlib import Path

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.event_common import check_result, get_origin_trace_id

_workload_build_path = os.path.join(
    get_buckyball_path(), "bb-tests", "workloads", "scripts", "build.py"
)
_spec = importlib.util.spec_from_file_location("workload_build_module", _workload_build_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_workload_build_path}")
workload_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workload_build)

config = {
    "name": "workload-build",
    "description": "build workload",
    "flows": ["workload"],
    "triggers": [queue("workload.build")],
    "enqueues": [],
}

# CLI model → layout dir under archs/buckyball/<chip>/
MODEL_LAYOUT = {
    "lenet": "LeNet",
    "mobilenet": "MobileNetV3",
    "resnet": "ResNet18",
    "yolo": "YOLO26",
    "bert": "Bert",
    "qwen3": "Qwen3",
    "gemma4": "Gemma4",
    "deepseekr1": "DeepSeekR1",
    "llama2": "llama2",
    "stable-diffusion": "StableDiffusion",
    "whisper": "Whisper",
    "buddynext": "BuddyNext",
}


def chips_for_model(bbdir: str, model_key: str) -> set[str]:
    """Chips that currently ship a layout for this model (many-to-many)."""
    layout = MODEL_LAYOUT.get(model_key)
    if layout is None:
        return set()
    root = (
        Path(bbdir)
        / "bb-tests"
        / "workloads"
        / "src"
        / "ModelTest"
        / "e2e"
        / "models"
        / "archs"
        / "buckyball"
    )
    if not root.is_dir():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / layout).is_dir()
    }


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
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

    if model:
        model_key = model.lower()
        layout = MODEL_LAYOUT.get(model_key)
        if layout is None:
            ctx.logger.error(f"Unknown model: {model}")
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "unknown_model", "model": model},
                trace_id=origin_tid,
            )
            return
        supported_chips = chips_for_model(bbdir, model_key)
        if chip not in supported_chips:
            allowed = ", ".join(sorted(supported_chips)) if supported_chips else "(none)"
            ctx.logger.error(
                f"Model '{model}' has no Buckyball layout on chip '{chip}' "
                f"(layout dir '{layout}'; chips with layout: {allowed})"
            )
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={
                    "error": "unsupported_chip_model",
                    "chip": chip,
                    "model": model,
                    "layout": layout,
                    "supported_chips": sorted(supported_chips),
                },
                trace_id=origin_tid,
            )
            return

    try:
        workload_build.build_workload(
            bbdir,
            chip,
            model=model.lower() if model else "",
            rushb=rushb_backend,
            stable=stable,
            logger=ctx.logger,
            task_scope=origin_tid,
        )
    except Exception as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "workload_build_failed", "chip": chip, "detail": str(error)},
            trace_id=origin_tid,
        )
        return

    await check_result(
        ctx, 0, continue_run=False,
        extra_fields={"chip": chip, "model": model, "rushB": rushb_backend},
        trace_id=origin_tid)

    return
