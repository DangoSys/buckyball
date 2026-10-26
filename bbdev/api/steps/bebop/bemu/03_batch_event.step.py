"""
bebop bemu batch event handler

Runs bebop bemu batch regression:
  1. Build the selected design's BEMU wrapper
  2. Run the BEMU regression harness
"""

from pathlib import Path
from utils.path import design_output_root


import os
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
from utils.path import (
    bebop_target_dir,
    bebop_bin_path,
    get_buckyball_path,
    workloads_output_root,
)
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from bemu_common import bemu_manifest
from utils.workload_manifest import resolve_workload_toml

config = {
    "name": "bebop-bemu-batch",
    "description": "Run bebop bemu batch regression",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.batch")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    design = input_data.get("design")
    if not design:
        ctx.logger.error("Missing required parameter: design must be specified")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_design"},
            trace_id=origin_tid,
        )
        return
    try:
        bemu_cargo_manifest = bemu_manifest(
            design, bbdir, selection=input_data["target"]
        )
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "invalid_design", "design": design},
            trace_id=origin_tid,
        )
        return
    elf_root = str(
        Path(design_output_root(bbdir, design, selection=input_data["target"]))
        / "workloads"
    )
    env = os.environ.copy()
    test_type = input_data.get("test", "bare-tests")
    try:
        workload_toml = resolve_workload_toml(
            design, "bemu", test_type, bbdir, selection=input_data["target"]
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
                "design": design,
            },
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"Running {test_type} with workload config: {workload_toml}")
    target = bebop_target_dir(bbdir, design, "bemu", selection=input_data["target"])
    build_cmd = (
        "nix develop -c cargo build --manifest-path"
        f" {shlex.quote(str(bemu_cargo_manifest))} --target-dir {shlex.quote(target)}"
        " --features bemu --tests --bin bebop-bemu"
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
            ctx,
            build_result.returncode,
            continue_run=False,
            extra_fields={"task": "build", "backend": "bemu"},
            trace_id=origin_tid,
        )
        return
    harness_args = [
        "--",
        "--workload-toml",
        workload_toml,
        "--bb-tests-root",
        elf_root,
        "--jobs",
        str(input_data.get("jobs", 1)),
    ]
    if input_data.get("clean-before", input_data.get("clean_before", False)):
        harness_args.append("--clean-before")
    if input_data.get("junit"):
        harness_args.extend(["--junit", input_data["junit"]])
    harness = shlex.join(harness_args)
    test_cmd = (
        "nix develop -c cargo test --manifest-path"
        f" {shlex.quote(str(bemu_cargo_manifest))} --target-dir {shlex.quote(target)}"
        f" --features bemu --test test_bemu {harness}"
    )
    ctx.logger.info(f"Running bebop bemu regression: {test_cmd}")
    run_result = await stream_run_logger_async(
        cmd=test_cmd,
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
            "design": design,
            "test_type": test_type,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
