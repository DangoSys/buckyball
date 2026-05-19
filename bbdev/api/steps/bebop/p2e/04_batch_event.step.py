"""
bebop p2e batch event handler

Runs bebop p2e nextest batch regression:
  1. Build bebop with p2e feature
  2. Run cargo nextest with p2e-specific config (serial, FPGA single-resource)
"""
import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-p2e-batch",
    "description": "Run bebop p2e nextest batch regression",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.batch")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    nextest_config = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/nextest.toml"
    elf_root = f"{bbdir}/bb-tests/output"

    bitstream = input_data.get("bitstream", "")
    build_dir = input_data.get("build_dir", "")
    if not bitstream or not build_dir:
        ctx.logger.error("Missing required parameters: bitstream and build_dir")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_params"},
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

    # ── Build bebop p2e (tests) ───────────────────────────────────────────
    build_cmd = "nix develop -c cargo build --features p2e --tests"
    ctx.logger.info("Building bebop p2e (tests)...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e build",
        stderr_prefix="bebop p2e build",
    )

    if build_result.returncode != 0:
        await check_result(
            ctx, build_result.returncode, continue_run=False,
            extra_fields={"task": "build", "backend": "p2e"},
            trace_id=origin_tid,
        )
        return

    # ── Run nextest ───────────────────────────────────────────────────────
    # Pass parameters via environment variables (nextest doesn't support custom CLI args after `--`)
    env = os.environ.copy()
    env.update({
        "BEBOP_WORKLOAD_TOML": workload_toml,
        "BEBOP_BB_TESTS_ROOT": elf_root,
        "BEBOP_P2E_BITSTREAM": bitstream,
        "BEBOP_P2E_BUILD_DIR": build_dir,
    })
    nextest_cmd = (
        f"nix develop -c cargo nextest run --features p2e --test test_p2e "
        f"--config-file \"{nextest_config}\""
    )

    ctx.logger.info(f"Running bebop p2e nextest: {nextest_cmd}")
    ctx.logger.info(f"Environment: BEBOP_WORKLOAD_TOML={workload_toml}, BEBOP_BB_TESTS_ROOT={elf_root}")
    run_result = stream_run_logger(
        cmd=nextest_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e batch",
        stderr_prefix="bebop p2e batch",
        env=env,
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "batch",
            "backend": "p2e",
            "bitstream": bitstream,
            "build_dir": build_dir,
            "test_type": test_type,
            "nextest_config": nextest_config,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
