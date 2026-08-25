"""
bebop verilator batch event handler

Runs bebop verilator nextest batch regression (requires prior --build).
"""
import os
import shutil
import shlex
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
bebop_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if bebop_path not in sys.path:
    sys.path.insert(0, bebop_path)

from utils.event_common import require_chip
from utils.path import bebop_cargo_env, chip_output_root, get_buckyball_path, rtl_dir
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from regression import regression_workload_toml
from regression_harness import nextest_harness_args


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

    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return

    elf_root = chip_output_root(bbdir, chip)

    test_type = input_data.get("test", "elf-tests")
    rushB = bool(input_data.get("rushB", False))
    diff = bool(input_data.get("diff", False))
    if diff and rushB:
        ctx.logger.error("--diff and --rushB cannot be used together")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "diff_conflicts_with_rushB"},
            trace_id=origin_tid,
        )
        return
    try:
        workload_toml = regression_workload_toml(
            chip, "verilator", test_type, bbdir, rushB=rushB, diff=diff
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
        f"Running {test_type} with workload config: {workload_toml} "
        f"rushB={rushB} diff={diff}"
    )

    vsrc_dir = rtl_dir(bbdir, chip, "verilog", input_data.get("vsrc_dir"))
    vsrc_config = shlex.quote(f"env.VSRC_PATH='{vsrc_dir}'")
    env = os.environ.copy()
    env.update(bebop_cargo_env(bbdir, chip))
    env.update({
        "BEBOP_ARCH_CONFIG": chip,
        "VSRC_PATH": vsrc_dir,
    })

    manifest = f"{bebop_dir}/Cargo.toml"
    features = "verilator"
    if diff:
        features = "verilator,bemu,difftest"
        preload = os.pathsep.join(
            path
            for path in (
                f"{bbdir}/result/lib/libdramsim3.so",
                os.environ.get("LD_PRELOAD", ""),
            )
            if path
        )
        env.update({
            "BEBOP_VERILATOR_DIFF": "1",
            "BEBOP_DIFF_LD_PRELOAD": preload,
            "BEBOP_DIFF_RUN_DIR": os.path.dirname(manifest),
        })

    if input_data.get("clean-before", input_data.get("clean_before", False)):
        artifact_dir = os.path.join(os.path.dirname(manifest), "test-artifacts")
        shutil.rmtree(artifact_dir, ignore_errors=True)
        ctx.logger.info(f"Cleaned previous bebop test artifacts: {artifact_dir}")

    harness = nextest_harness_args(workload_toml, elf_root)
    nextest_cmd = (
        f"nix develop -c cargo nextest run --release --manifest-path {shlex.quote(manifest)} "
        f"--features {shlex.quote(features)} --test test_verilator "
        f"--config-file {shlex.quote(nextest_config)} "
        f"--config={vsrc_config} "
        f"{harness}"
    )

    ctx.logger.info(f"Running bebop verilator nextest: {nextest_cmd}")
    run_result = await stream_run_logger_async(
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
            "backend": "verilator-rushB" if rushB else ("difftest" if diff else "verilator"),
            "chip": chip,
            "vsrc_dir": vsrc_dir,
            "test_type": test_type,
            "rushB": rushB,
            "diff": diff,
            "nextest_config": nextest_config,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
