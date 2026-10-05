"""
bebop verilator batch event handler

Runs bebop verilator batch regression (requires prior --build).
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
from utils.event_common import require_chip
from utils.path import (
    bebop_cargo_env,
    get_buckyball_path,
    rtl_dir,
    workloads_output_root,
)
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from utils.workload_manifest import resolve_workload_toml

config = {
    "name": "bebop-verilator-batch",
    "description": "Run bebop verilator batch regression",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.batch")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    elf_root = workloads_output_root(bbdir)
    test_type = input_data.get("test", "bare-tests")
    diff = bool(input_data.get("diff", False))
    try:
        workload_toml = resolve_workload_toml(
            chip, "verilator", test_type, bbdir, diff=diff
        )
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "error": "invalid_regression",
                "test": test_type,
                "chip": chip,
            },
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(
        f"Running {test_type} with workload config: {workload_toml} diff={diff}"
    )
    vsrc_dir = rtl_dir(bbdir, chip, "verilog", input_data.get("vsrc_dir"))
    vsrc_config = shlex.quote(f"env.VSRC_PATH='{vsrc_dir}'")
    env = os.environ.copy()
    env.update(bebop_cargo_env(bbdir, chip))
    env.update({"VSRC_PATH": vsrc_dir})
    manifest = f"{bebop_dir}/Cargo.toml"
    features = "verilator"
    if diff:
        manifest = f"{bbdir}/examples/chips/{chip}/generated/bebop/Cargo.toml"
        features = "verilator,bemu"
    if input_data.get("clean-before", input_data.get("clean_before", False)):
        artifact_dir = os.path.join(env["CARGO_TARGET_DIR"], "test-artifacts")
        shutil.rmtree(artifact_dir, ignore_errors=True)
        ctx.logger.info(f"Cleaned previous bebop test artifacts: {artifact_dir}")
    harness_args = [
        "--",
        "--workload-toml",
        workload_toml,
        "--bb-tests-root",
        elf_root,
        "--arch-config",
        chip,
        "--jobs",
        str(input_data.get("jobs", 1)),
    ]
    if diff:
        harness_args.append("--diff")
    harness = shlex.join(harness_args)
    test_cmd = f"nix develop -c cargo --config={vsrc_config} test --release --manifest-path {shlex.quote(manifest)} --features {shlex.quote(features)} --test test_verilator {harness}"
    ctx.logger.info(f"Running bebop verilator regression: {test_cmd}")
    run_result = await stream_run_logger_async(
        cmd=test_cmd,
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
            "backend": "difftest" if diff else "verilator",
            "chip": chip,
            "vsrc_dir": vsrc_dir,
            "test_type": test_type,
            "diff": diff,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
