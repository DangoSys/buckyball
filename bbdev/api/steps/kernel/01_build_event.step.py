from utils.path import design_resource_dir
import os
import shlex
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import design_output_root, get_buckyball_path, load_design_config
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "kernel-build",
    "description": "build RISC-V kernel + rootfs for image via stack/os/linux",
    "flows": ["kernel"],
    "triggers": [queue("kernel.build")],
    "enqueues": [],
}


def kernel_build_dir(
    bbdir: str,
    hart_count: int,
    design: str,
    selection: str,
    interactive: bool,
    guest_memory_mib: int,
) -> str:
    suffix = "" if guest_memory_mib == 512 else f"-mem{guest_memory_mib}M"
    suffix += f"-design-{design}-{selection.replace('/', '-')}"
    if interactive:
        suffix += "-interactive"
    name = "kernel" if hart_count == 64 else f"kernel-h{hart_count}"
    return os.path.join(bbdir, "bb-tests", "build", f"{name}{suffix}")


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    kernel_src = os.path.join(bbdir, "stack", "os", "linux")
    design = input_data["design"]
    design_config = load_design_config(bbdir, design, selection=input_data["target"])
    hart_count = len(design_config.device.cores)
    ddr = design_config.device.ddr
    guest_memory_mib = (ddr.bytes - ddr.model) // (1024 * 1024)
    # A DDR without a model region leaves the base empty: init runs no model.
    model_base = f"{ddr.base + ddr.bytes - ddr.model:#x}" if ddr.model else ""
    interactive = input_data.get("interactive", False)
    output_root = design_output_root(bbdir, design, selection=input_data["target"])
    output_dir = os.path.join(output_root, "workloads/kernel")
    os.makedirs(output_dir, exist_ok=True)
    kernel_build = kernel_build_dir(
        bbdir, hart_count, design, input_data["target"], interactive, guest_memory_mib
    )
    interactive_arg = "ON" if interactive else "OFF"
    programs = design_resource_dir(
        bbdir, design, "kernel", selection=input_data["target"]
    )

    # cmake configure
    configure_cmd = (
        f"cmake -B {kernel_build} -S {kernel_src} "
        f"-DBUCKYBALL_HART_COUNT={hart_count} "
        f"-DBUCKYBALL_GUEST_MEMORY_MIB={guest_memory_mib} "
        f"-DBUCKYBALL_KERNEL_DESIGN={design} "
        f"-DBUCKYBALL_KERNEL_OUTPUT={output_root} "
        f"-DBUCKYBALL_KERNEL_INTERACTIVE={interactive_arg} "
        f"-DBUCKYBALL_MODEL_BASE={model_base} "
        f"-DBUCKYBALL_KERNEL_WORKLOADS={programs}"
    )

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

    # The output directory belongs to one device config, so the name is fixed.
    payload_name = "fw_payload-linux"
    fw_payload_bin = os.path.join(output_dir, f"{payload_name}.bin")
    fw_payload_elf = os.path.join(output_dir, f"{payload_name}.elf")
    built_payload_elf = os.path.join(
        kernel_build,
        "opensbi",
        "platform",
        "buckyball",
        "firmware",
        "fw_payload.elf",
    )

    shutil.copy2(built_payload_elf, fw_payload_elf)

    # The DDR image: the kernel and, when given, the model, each at its place.
    # BEMU takes the bin, P2E the hex.
    image = os.path.join(output_dir, f"{payload_name}-ddr")
    pack = os.path.join(kernel_build, "image-pack")
    model = input_data.get("model")
    pack_cmd = [pack, f"{image}.bin", f"{image}.hex", str(ddr.bytes), fw_payload_bin]
    if model:
        pack_cmd += [
            os.path.join(
                bbdir, "stack/models/archs/buckyball", design, model, "output/model.img"
            ),
            str(ddr.bytes - ddr.model),
        ]
    result = await stream_run_logger_async(
        cmd=shlex.join(pack_cmd),
        logger=ctx.logger,
        stdout_prefix="image pack",
        stderr_prefix="image pack",
    )
    await check_result(ctx, result.returncode, continue_run=False, trace_id=origin_tid)
