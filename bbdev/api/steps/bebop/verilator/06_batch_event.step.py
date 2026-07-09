"""
bebop verilator batch event handler

Runs bebop verilator nextest batch regression:
  1. Build bebop with verilator feature and VSRC_PATH
  2. Run cargo nextest with verilator-specific config
"""
import os
import shutil
import shlex
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-verilator-batch",
    "description": "Run bebop verilator nextest batch regression",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.batch")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    nextest_config = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/nextest.toml"
    elf_root = f"{bbdir}/bb-tests/output"

    arch_config = input_data.get("config")
    if not isinstance(arch_config, str) or not arch_config or arch_config == "None":
        ctx.logger.error("Missing required parameter: config must be specified")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_config"},
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

    vsrc_dir = get_verilator_build_dir(bbdir, arch_config, input_data.get("vsrc_dir"))
    if not os.path.isdir(vsrc_dir):
        ctx.logger.error(f"VSRC_PATH does not exist: {vsrc_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "vsrc_not_found", "vsrc_dir": vsrc_dir},
            trace_id=origin_tid,
        )
        return

    # ── Build bebop verilator (tests) ─────────────────────────────────────
    vsrc_config = shlex.quote(f"env.VSRC_PATH='{vsrc_dir}'")
    build_cmd = (
        f"nix develop -c cargo build --manifest-path {shlex.quote(f'{bebop_dir}/Cargo.toml')} "
        "--features verilator --tests "
        f"--config={vsrc_config}"
    )
    ctx.logger.info("Building bebop verilator (tests)...")
    build_result = stream_run_logger(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop verilator build",
        stderr_prefix="bebop verilator build",
    )

    if build_result.returncode != 0:
        await check_result(
            ctx, build_result.returncode, continue_run=False,
            extra_fields={"task": "build", "backend": "verilator", "vsrc_dir": vsrc_dir},
            trace_id=origin_tid,
        )
        return

    # ── Run nextest ───────────────────────────────────────────────────────
    # Pass parameters via environment variables (nextest doesn't support custom CLI args after `--`)
    if input_data.get("clean-before", input_data.get("clean_before", False)):
        shutil.rmtree(f"{bebop_dir}/test-artifacts", ignore_errors=True)
        ctx.logger.info("Cleaned previous bebop test artifacts")

    env = os.environ.copy()
    env.update({
        "BEBOP_WORKLOAD_TOML": workload_toml,
        "BEBOP_BB_TESTS_ROOT": elf_root,
        "VSRC_PATH": vsrc_dir,
    })
    nextest_cmd = (
        f"nix develop -c cargo nextest run --manifest-path {shlex.quote(f'{bebop_dir}/Cargo.toml')} "
        "--features verilator --test test_verilator "
        f"--config-file {shlex.quote(nextest_config)} "
        f"--config={vsrc_config}"
    )

    ctx.logger.info(f"Running bebop verilator nextest: {nextest_cmd}")
    ctx.logger.info(f"Environment: BEBOP_WORKLOAD_TOML={workload_toml}, BEBOP_BB_TESTS_ROOT={elf_root}")
    run_result = stream_run_logger(
        cmd=nextest_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop verilator batch",
        stderr_prefix="bebop verilator batch",
        env=env,
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "batch",
            "backend": "verilator",
            "config": arch_config,
            "vsrc_dir": vsrc_dir,
            "test_type": test_type,
            "nextest_config": nextest_config,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
