import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

# Import bin_to_hex converter
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
from bin_to_hex import bin_to_hex

KERNEL_MODELS = {
    "bert",
    "deepseekr1",
    "gemma4",
    "lenet",
    "llama2",
    "mobilenet",
    "qwen3",
    "resnet",
    "stablediffusion",
    "yolo",
}

config = {
    "name": "kernel-build",
    "description": "build RISC-V kernel + rootfs for image via bb-tests/workloads/lib/kernel",
    "flows": ["kernel"],
    "triggers": [queue("kernel.build")],
    "enqueues": [],
}


def hart_count_params(input_data: dict) -> dict:
    allowed = {"visible-hart-count", "total-hart-count", "model", "_trace_id"}
    unknown = sorted(k for k in input_data if k not in allowed)
    if unknown:
        raise ValueError(f"unknown kernel build parameter(s): {', '.join(unknown)}")

    if "hidden-hart-base" in input_data:
        raise ValueError("hidden-hart-base is not supported; hidden harts must start at visible-hart-count")

    visible = int(input_data.get("visible-hart-count", 64))
    total = int(input_data.get("total-hart-count", visible))
    hidden_base = visible

    if visible < 1:
        raise ValueError("visible-hart-count must be at least 1")
    if total < visible:
        raise ValueError("total-hart-count must cover visible harts")

    return {
        "visible": visible,
        "total": total,
        "hidden_base": hidden_base,
    }


def kernel_model(input_data: dict) -> str:
    model = input_data.get("model", "")
    if model in ("", None):
        return ""
    if not isinstance(model, str):
        raise ValueError("model must be a string")

    model = model.lower()
    if model not in KERNEL_MODELS:
        valid = ", ".join(sorted(KERNEL_MODELS))
        raise ValueError(f"unknown kernel model: {model}; valid models: {valid}")
    return model


def kernel_build_dir(bbdir: str, hart_params: dict, model: str = "") -> str:
    visible = hart_params["visible"]
    total = hart_params["total"]
    model_suffix = f"-model-{model}" if model else ""
    if visible == 64 and total == 64:
        return os.path.join(bbdir, "bb-tests", "build", f"kernel{model_suffix}")
    return os.path.join(bbdir, "bb-tests", "build", f"kernel-v{visible}-t{total}{model_suffix}")


def fw_payload_name(hart_params: dict) -> str:
    visible = hart_params["visible"]
    total = hart_params["total"]
    if visible == 64 and total == 64:
        return "fw_payload"
    return f"fw_payload-v{visible}-t{total}"


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    kernel_src = os.path.join(bbdir, "bb-tests", "workloads", "lib", "kernel")
    output_dir = os.path.join(bbdir, "bb-tests", "output", "kernel")

    os.makedirs(output_dir, exist_ok=True)

    try:
        hart_params = hart_count_params(input_data)
        model = kernel_model(input_data)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return
    kernel_build = kernel_build_dir(bbdir, hart_params, model)

    # cmake configure
    configure_cmd = (
        f"cmake -B {kernel_build} -S {kernel_src} "
        f"-DBUCKYBALL_VISIBLE_HART_COUNT={hart_params['visible']} "
        f"-DBUCKYBALL_TOTAL_HART_COUNT={hart_params['total']} "
        f"-DBUCKYBALL_HIDDEN_HART_BASE={hart_params['hidden_base']} "
        f"-DBUCKYBALL_KERNEL_MODEL={model}"
    )
    result = stream_run_logger(
        cmd=configure_cmd,
        logger=ctx.logger,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )
    if result.returncode != 0:
        await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)
        return

    # cmake build
    build_cmd = f"cmake --build {kernel_build} --target kernel-build"
    result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )

    if result.returncode != 0:
        await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)
        return

    # Convert fw_payload.bin to hex for P2E memory backdoor
    payload_name = fw_payload_name(hart_params)
    fw_payload_bin = os.path.join(output_dir, f"{payload_name}.bin")
    fw_payload_hex = os.path.join(output_dir, f"{payload_name}.hex")

    if not os.path.exists(fw_payload_bin):
        ctx.logger.error(f"{payload_name}.bin not found")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    ctx.logger.info(f"Converting {fw_payload_bin} to Verilog hex format for P2E...")
    success = bin_to_hex(fw_payload_bin, fw_payload_hex, base_address=0x80000000)
    if not success:
        ctx.logger.error("Failed to convert fw_payload to hex")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)
