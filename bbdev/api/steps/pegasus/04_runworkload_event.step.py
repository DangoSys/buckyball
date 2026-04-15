"""
pegasus runworkload event handler

Sequence:
  1. Resolve marshal image paths (kernel ELF + rootfs img)
  2. Build pegasus-driver if needed
  3. pegasus-driver load --kernel <elf> --rootfs <img>
  4. pegasus-driver run --log <log_path> --timeout <t>
"""
import os
import sys
import datetime
import subprocess

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "pegasus-runworkload",
    "description": "Load Linux image into HBM2 and run on AU280",
    "flows": ["pegasus"],
    "triggers": [queue("pegasus.runworkload")],
    "enqueues": [],
}

KERNEL_OUTPUT_DIR = "bb-tests/output/kernel"


def _resolve_images(bbdir: str, workload: str, board: str) -> tuple[str, str]:
    """Return (kernel_path, rootfs_path) from bbkernel output."""
    output_dir = os.path.join(bbdir, KERNEL_OUTPUT_DIR)
    kernel = os.path.join(output_dir, "pegasus-bin")
    rootfs = os.path.join(output_dir, "pegasus.img")
    return kernel, rootfs


def _build_driver(bbdir: str, logger) -> str | None:
    """Build pegasus-driver if not already built. Returns binary path or None."""
    driver_src = os.path.join(bbdir, "thirdparty", "pegasus", "driver")
    build_dir  = os.path.join(driver_src, "build")
    binary     = os.path.join(build_dir, "pegasus-driver")

    if os.path.exists(binary):
        logger.info(f" driver already built: {binary}")
        return binary

    logger.info(" building pegasus-driver ...")
    os.makedirs(build_dir, exist_ok=True)

    # Ensure pkg-config can find nix-managed libraries (libelf from elfutils)
    result_dir = os.path.join(bbdir, "result")
    pkg_config_path = os.environ.get("PKG_CONFIG_PATH", "")
    nix_pc = os.path.join(result_dir, "lib", "pkgconfig")
    if nix_pc not in pkg_config_path:
        pkg_config_path = f"{nix_pc}:{pkg_config_path}" if pkg_config_path else nix_pc

    cmake_result = stream_run_logger(
        cmd="cmake .. -DCMAKE_BUILD_TYPE=Release",
        logger=logger,
        cwd=build_dir,
        stdout_prefix="cmake",
        stderr_prefix="cmake",
        env={**os.environ, "PKG_CONFIG_PATH": pkg_config_path},
    )
    if cmake_result.returncode != 0:
        logger.error(" cmake configure failed")
        return None

    build_result = stream_run_logger(
        cmd=f"cmake --build . -j$(nproc)",
        logger=logger,
        cwd=build_dir,
        stdout_prefix="make",
        stderr_prefix="make",
    )
    if build_result.returncode != 0:
        logger.error(" driver build failed")
        return None

    logger.info(f" driver built: {binary}")
    return binary


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    workload    = input_data.get("workload", "interactive")
    board       = input_data.get("board",    "chipyard")
    timeout     = int(input_data.get("timeout", 300))
    uart_dev    = input_data.get("uart",     "/dev/ttyUSB0")
    control_dev = input_data.get("control",  "/dev/xdma0_user")
    h2c_dev     = input_data.get("h2c",      "/dev/xdma0_h2c_0")

    # Log output directory
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(bbdir, "arch", "log", ts)
    os.makedirs(log_dir, exist_ok=True)
    uart_log = os.path.join(log_dir, "pegasus_uart.log")

    # ── 1. Resolve image paths ────────────────────────────────────────────
    kernel, rootfs = _resolve_images(bbdir, workload, board)
    ctx.logger.info(f" kernel: {kernel}")
    ctx.logger.info(f" rootfs: {rootfs}")

    for path in (kernel, rootfs):
        if not os.path.exists(path):
            ctx.logger.error(f" image not found: {path} — run 'bbdev kernel --build' first")
            await check_result(ctx, 1, continue_run=False,
                               extra_fields={"error": "image_not_found", "path": path},
                               trace_id=origin_tid)
            return

    # ── 2. Build driver ───────────────────────────────────────────────────
    driver = _build_driver(bbdir, ctx.logger)
    if driver is None:
        await check_result(ctx, 1, continue_run=False,
                           extra_fields={"error": "driver_build_failed"},
                           trace_id=origin_tid)
        return

    # ── 3. Load image into HBM2 ───────────────────────────────────────────
    load_cmd = (
        f"sudo {driver} load"
        f" --kernel {kernel}"
        f" --rootfs {rootfs}"
        f" --h2c {h2c_dev}"
    )
    ctx.logger.info(" loading images into HBM2 ...")
    load_result = stream_run_logger(
        cmd=load_cmd,
        logger=ctx.logger,
        stdout_prefix="pegasus load",
        stderr_prefix="pegasus load",
    )
    if load_result.returncode != 0:
        await check_result(ctx, load_result.returncode, continue_run=False,
                           extra_fields={"error": "load_failed"},
                           trace_id=origin_tid)
        return

    # ── 4. Start CPU and collect UART ─────────────────────────────────────
    run_cmd = (
        f"sudo {driver} run"
        f" --control {control_dev}"
        f" --uart {uart_dev}"
        f" --log {uart_log}"
        f" --timeout {timeout}"
    )
    ctx.logger.info(f" starting CPU, collecting UART → {uart_log}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        stdout_prefix="pegasus uart",
        stderr_prefix="pegasus uart",
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "workload":  workload,
            "board":     board,
            "uart_log":  uart_log,
        },
        trace_id=origin_tid,
    )
