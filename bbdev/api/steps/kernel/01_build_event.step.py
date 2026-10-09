import os
import json
import re
import shutil
import sys
import tomllib

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

# Import bin_to_hex converter
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
from bin_to_hex import bin_to_hex

config = {
    "name": "kernel-build",
    "description": "build RISC-V kernel + rootfs for image via bb-tests/workloads/lib/kernel",
    "flows": ["kernel"],
    "triggers": [queue("kernel.build")],
    "enqueues": [],
}


def hart_count_params(input_data: dict) -> dict:
    allowed = {
        "hart-count",
        "model",
        "chip",
        "interactive",
        "guest-memory-mib",
        "model-storage",
        "_trace_id",
    }
    unknown = sorted(k for k in input_data if k not in allowed)
    if unknown:
        raise ValueError(f"unknown kernel build parameter(s): {', '.join(unknown)}")

    if "hart-count" in input_data:
        value = input_data["hart-count"]
        if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
            raise ValueError("hart-count must be a positive integer")

    chip = input_data.get("chip")
    if chip:
        derived = os.path.join(
            get_buckyball_path(),
            "examples",
            "chips",
            chip,
            "configs",
            "generated",
            "config",
            "derived.json",
        )
        with open(derived) as stream:
            harts = json.load(stream)["harts"]
        count = sum(hart["visible"] for hart in harts)
        if "hart-count" in input_data and int(input_data["hart-count"]) != count:
            raise ValueError(f"hart-count must match chip topology: {count}")
    else:
        count = int(input_data.get("hart-count", 64))

    if count < 1:
        raise ValueError("hart-count must be at least 1")

    return {"count": count}


def kernel_interactive(input_data: dict) -> bool:
    interactive = input_data.get("interactive", False)
    if not isinstance(interactive, bool):
        raise ValueError("interactive must be a boolean flag")
    return interactive


def kernel_model(input_data: dict) -> str:
    model = input_data.get("model", "")
    if model in ("", None):
        return ""
    if not isinstance(model, str):
        raise ValueError("model must be a string")

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", model):
        raise ValueError(f"invalid model name: {model}")
    return model


def kernel_chip(input_data: dict, bbdir: str) -> str:
    chip = input_data.get("chip", "")
    if chip in ("", None):
        return ""
    if not isinstance(chip, str):
        raise ValueError("chip must be a string")
    if os.path.sep in chip or chip in {".", ".."}:
        raise ValueError(f"invalid chip: {chip}")

    chip_dir = os.path.join(bbdir, "examples", "chips", chip)
    if not os.path.isdir(chip_dir):
        raise ValueError(f"unknown chip: {chip}")

    return chip


def kernel_build_dir(
    bbdir: str,
    hart_params: dict,
    model: str = "",
    chip: str = "",
    interactive: bool = False,
    guest_memory_mib: int = 512,
    model_storage: str = "initramfs",
) -> str:
    count = hart_params["count"]
    suffix = "" if guest_memory_mib == 512 else f"-mem{guest_memory_mib}M"
    if chip:
        suffix += f"-chip-{chip}"
    if model:
        suffix += f"-model-{model}"
    if interactive:
        suffix += "-interactive"
    if model_storage == "ddr":
        suffix += "-ddr"
    if count == 64:
        return os.path.join(bbdir, "bb-tests", "build", f"kernel{suffix}")
    return os.path.join(
        bbdir, "bb-tests", "build", f"kernel-h{count}{suffix}"
    )


def fw_payload_name(
    hart_params: dict,
    model: str = "",
    chip: str = "",
    guest_memory_mib: int = 512,
    model_storage: str = "initramfs",
) -> str:
    count = hart_params["count"]
    name = "fw_payload"
    if count != 64:
        name = f"{name}-h{count}"
    if model:
        name = f"{name}-{model}"
    elif chip:
        name = f"{name}-{chip}-linux"
    if guest_memory_mib != 512:
        name += f"-mem{guest_memory_mib}M"
    if model_storage == "ddr":
        name += "-ddr"
    return name


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    kernel_src = os.path.join(bbdir, "bb-tests", "workloads", "lib", "kernel")
    try:
        model = kernel_model(input_data)
        if model and not input_data.get("chip"):
            raise ValueError("--model requires --chip")
        chip = kernel_chip(input_data, bbdir)
        memory_value = input_data.get("guest-memory-mib", 512)
        if model and "guest-memory-mib" not in input_data:
            config_path = os.path.join(
                bbdir, "examples", "models", chip, model, "configs", "running-param.toml"
            )
            with open(config_path, "rb") as stream:
                memory_value = tomllib.load(stream).get("memory_mib", 512)
        if (
            isinstance(memory_value, bool)
            or not str(memory_value).isdigit()
            or not 1 <= int(memory_value) <= 16384
        ):
            raise ValueError(
                "guest-memory-mib must be an integer in 1..16384 (P2E DDR capacity)"
            )
        guest_memory_mib = int(memory_value)
        model_storage = input_data.get("model-storage", "initramfs")
        if model_storage not in ("initramfs", "ddr"):
            raise ValueError("model-storage must be initramfs or ddr")
        if model_storage == "ddr" and (
            not input_data.get("model") or guest_memory_mib >= 16384
        ):
            raise ValueError("ddr requires --model and guest-memory-mib < 16384")
        hart_params = hart_count_params(input_data)
        interactive = kernel_interactive(input_data)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return
    output_dir = os.path.join(bbdir, "bb-tests", "output", "kernel")
    if chip:
        output_dir = os.path.join(output_dir, chip)
    os.makedirs(output_dir, exist_ok=True)
    kernel_build = kernel_build_dir(
        bbdir,
        hart_params,
        model,
        chip,
        interactive=interactive,
        guest_memory_mib=guest_memory_mib,
        model_storage=model_storage,
    )
    interactive_arg = "ON" if interactive else "OFF"
    cmake_model_name = model

    # cmake configure
    configure_cmd = (
        f"cmake -B {kernel_build} -S {kernel_src} "
        f"-DBUCKYBALL_HART_COUNT={hart_params['count']} "
        f"-DBUCKYBALL_GUEST_MEMORY_MIB={guest_memory_mib} "
        f"-DBUCKYBALL_MODEL_STORAGE={model_storage} "
        f"-DBUCKYBALL_KERNEL_MODEL={cmake_model_name} "
        f"-DBUCKYBALL_KERNEL_CHIP={chip} "
        f"-DBUCKYBALL_KERNEL_INTERACTIVE={interactive_arg} "
        f"-DBUCKYBALL_MODEL_DATASET="
    )
    if chip and not model:
        workload_toml = os.path.join(
            bbdir, "examples", "chips", chip, "kernel", "workloads.toml"
        )
        if not os.path.isfile(workload_toml):
            ctx.logger.error(f"chip workload toml not found: {workload_toml}")
            await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
            return
        configure_cmd += f" -DBUCKYBALL_WORKLOAD_TOML={workload_toml}"

    result = await stream_run_logger_async(
        cmd=configure_cmd,
        logger=ctx.logger,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )
    if result.returncode != 0:
        await check_result(
            ctx, result.returncode, continue_run=False, trace_id=origin_tid
        )
        return

    # cmake build
    build_cmd = f"cmake --build {kernel_build} --target kernel-build"
    result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        stdout_prefix="marshal build",
        stderr_prefix="marshal build",
    )

    if result.returncode != 0:
        await check_result(
            ctx, result.returncode, continue_run=False, trace_id=origin_tid
        )
        return

    # Convert fw_payload.bin to hex for P2E memory backdoor
    payload_name = fw_payload_name(
        hart_params,
        cmake_model_name if model else "",
        chip,
        guest_memory_mib,
        model_storage,
    )
    fw_payload_bin = os.path.join(output_dir, f"{payload_name}.bin")
    fw_payload_hex = os.path.join(output_dir, f"{payload_name}.hex")
    fw_payload_elf = os.path.join(output_dir, f"{payload_name}.elf")
    built_payload_elf = os.path.join(
        kernel_build,
        "opensbi",
        "platform",
        "buckyball",
        "firmware",
        "fw_payload.elf",
    )

    if not os.path.exists(fw_payload_bin):
        ctx.logger.error(f"{payload_name}.bin not found")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return
    if not os.path.isfile(built_payload_elf):
        ctx.logger.error(f"fw_payload.elf not found: {built_payload_elf}")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    shutil.copy2(built_payload_elf, fw_payload_elf)

    ctx.logger.info(f"Converting {fw_payload_bin} to Verilog hex format for P2E...")
    success = bin_to_hex(fw_payload_bin, fw_payload_hex, base_address=0x80000000)
    if not success:
        ctx.logger.error("Failed to convert fw_payload to hex")
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    storage_fields = {"model_storage": model_storage}
    if model_storage == "ddr":
        manifest = os.path.join(output_dir, f"{payload_name}.load.json")
        if not os.path.isfile(manifest):
            raise ValueError(f"Missing ddr load manifest: {manifest}")
        storage_fields["load_manifest"] = manifest
    await check_result(
        ctx, 0, continue_run=False, extra_fields=storage_fields, trace_id=origin_tid
    )
