"""
bebop bemu batch event handler

Runs bebop bemu nextest batch regression:
  1. Build the selected chip's BEMU wrapper
  2. Run cargo nextest with bemu-specific config
"""
import os
import shutil
import shlex
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
bebop_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if bebop_path not in sys.path:
    sys.path.insert(0, bebop_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import bebop_cargo_env, chip_output_root, get_buckyball_path
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from bemu_common import bemu_manifest
from regression import regression_workload_toml
from regression_harness import nextest_harness_args


config = {
    "name": "bebop-bemu-batch",
    "description": "Run bebop bemu nextest batch regression",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.batch")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    nextest_config = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/nextest.toml"

    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip must be specified")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    try:
        bemu_cargo_manifest = bemu_manifest(chip, bbdir)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return
    elf_root = chip_output_root(bbdir, chip)

    env = os.environ.copy()
    env.update(bebop_cargo_env(bbdir, chip))
    test_type = input_data.get("test", "elf-tests")
    rushB = bool(input_data.get("rushB", False))
    try:
        workload_toml = regression_workload_toml(
            chip, "bemu", test_type, bbdir, rushB=rushB
        )
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_regression", "test": test_type, "chip": chip},
            trace_id=origin_tid,
        )
        return

    ctx.logger.info(
        f"Running {test_type} with workload config: {workload_toml} rushB={rushB}"
    )

    # ── Build bebop bemu ──────────────────────────────────────────────────
    build_cmd = (
        f"nix develop -c cargo build --manifest-path {shlex.quote(str(bemu_cargo_manifest))} "
        "--tests"
    )
    ctx.logger.info("Building bebop bemu (tests)...")
    build_result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop bemu build",
        stderr_prefix="bebop bemu build",
        env=env,
    )

    if build_result.returncode != 0:
        await check_result(
            ctx, build_result.returncode, continue_run=False,
            extra_fields={"task": "build", "backend": "bemu"},
            trace_id=origin_tid,
        )
        return

    # ── Run nextest ───────────────────────────────────────────────────────
    # Pass test harness parameters through nextest's process environment.
    if input_data.get("clean-before", input_data.get("clean_before", False)):
        shutil.rmtree(bemu_cargo_manifest.parent / "test-artifacts", ignore_errors=True)
        ctx.logger.info("Cleaned previous bebop test artifacts")

    harness = nextest_harness_args(workload_toml, elf_root)
    nextest_cmd = (
        f"nix develop -c cargo nextest run --manifest-path {shlex.quote(str(bemu_cargo_manifest))} "
        "--test test_bemu "
        f"--config-file {shlex.quote(nextest_config)} "
        f"{harness}"
    )

    ctx.logger.info(f"Running bebop bemu nextest: {nextest_cmd}")
    run_result = await stream_run_logger_async(
        cmd=nextest_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop bemu batch",
        stderr_prefix="bebop bemu batch",
        env=env,
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "batch",
            "backend": "bemu-rushB" if rushB else "bemu",
            "chip": chip,
            "test_type": test_type,
            "rushB": rushB,
            "nextest_config": nextest_config,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
