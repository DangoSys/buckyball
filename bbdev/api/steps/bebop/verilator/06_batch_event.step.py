"""
bebop verilator batch event handler

Runs bebop verilator batch regression (requires prior --build).
"""

from pathlib import Path
from utils.path import design_output_root
from utils.path import runner_manifest


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
from utils.event_common import require_design
from utils.path import (
    bebop_target_dir,
    get_buckyball_path,
    rtl_dir,
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
    try:
        design = require_design(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_design"},
            trace_id=origin_tid,
        )
        return
    elf_root = str(
        Path(design_output_root(bbdir, design, selection=input_data["target"]))
        / "workloads"
    )
    test_type = input_data.get("test", "bare-tests")
    diff = bool(input_data.get("diff", False))
    try:
        workload_toml = resolve_workload_toml(
            design,
            "verilator",
            test_type,
            bbdir,
            diff=diff,
            selection=input_data["target"],
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
    ctx.logger.info(
        f"Running {test_type} with workload config: {workload_toml} diff={diff}"
    )
    vsrc_dir = rtl_dir(
        bbdir,
        design,
        "verilog",
        selection=input_data["target"],
    )
    manifest = str(runner_manifest(bbdir, design, input_data["target"]))
    features = "verilator,bemu" if diff else "verilator"
    harness_args = [
        "--",
        "--workload-toml",
        workload_toml,
        "--bb-tests-root",
        elf_root,
        "--arch-config",
        design,
        "--jobs",
        str(input_data.get("jobs", 1)),
    ]
    if input_data.get("clean-before", input_data.get("clean_before", False)):
        harness_args.append("--clean-before")
    if diff:
        harness_args.append("--diff")
    if input_data.get("junit"):
        harness_args.extend(["--junit", input_data["junit"]])
    harness = shlex.join(harness_args)
    test_cmd = (
        "nix develop -c cargo test --release --target-dir"
        f" {shlex.quote(bebop_target_dir(bbdir, design, 'verilator', selection=input_data['target']))}"
        f" --manifest-path {shlex.quote(manifest)} --features {shlex.quote(features)}"
        f" --test test_verilator {harness}"
    )
    ctx.logger.info(f"Running bebop verilator regression: {test_cmd}")
    run_result = await stream_run_logger_async(
        cmd=test_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop verilator batch",
        stderr_prefix="bebop verilator batch",
    )
    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "batch",
            "backend": "difftest" if diff else "verilator",
            "design": design,
            "vsrc_dir": vsrc_dir,
            "test_type": test_type,
            "diff": diff,
            "workload_toml": workload_toml,
        },
        trace_id=origin_tid,
    )
