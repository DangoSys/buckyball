"""
bebop bemu batch event handler

Runs bebop bemu nextest batch regression:
  1. Build bebop with bemu feature
  2. Run cargo nextest with bemu-specific config
"""
import os
import shlex
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id
from bemu_common import bemu_env

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
    bebop_dir = f"{bbdir}/bebop"
    nextest_config = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/nextest.toml"
    elf_root = f"{bbdir}/bb-tests/output"

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
        env = os.environ.copy()
        env.update(bemu_env(chip, bbdir))
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return

    # ── Determine workload file based on test type ────────────────────────
    test_type = input_data.get("test", "elf-tests")
    if test_type == "elf-tests":
        workload_toml = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/workloads-elf.toml"
    elif test_type == "pk-tests":
        workload_toml = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/workloads-pk.toml"
    else:
        ctx.logger.error(f"Invalid test type: {test_type}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_test_type", "test": test_type},
            trace_id=origin_tid,
        )
        return

    ctx.logger.info(f"Running {test_type} with workload config: {workload_toml}")

    # ── Build bebop bemu ──────────────────────────────────────────────────
    build_cmd = (
        f"nix develop -c cargo build --manifest-path {shlex.quote(f'{bebop_dir}/Cargo.toml')} "
        "--features bemu --tests"
    )
    ctx.logger.info("Building bebop bemu (tests)...")
    build_result = stream_run_logger(
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
    # Pass parameters via environment variables (nextest doesn't support custom CLI args after `--`)
    env.update({
        "BEBOP_WORKLOAD_TOML": workload_toml,
        "BEBOP_BB_TESTS_ROOT": elf_root,
    })
    nextest_cmd = (
        f"nix develop -c cargo nextest run --manifest-path {shlex.quote(f'{bebop_dir}/Cargo.toml')} "
        "--features bemu --test test_bemu "
        f"--config-file {shlex.quote(nextest_config)}"
    )

    ctx.logger.info(f"Running bebop bemu nextest: {nextest_cmd}")
    ctx.logger.info(f"Environment: BEBOP_WORKLOAD_TOML={workload_toml}, BEBOP_BB_TESTS_ROOT={elf_root}")
    run_result = stream_run_logger(
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
            "backend": "bemu",
            "chip": chip,
            "test_type": test_type,
            "nextest_config": nextest_config,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
