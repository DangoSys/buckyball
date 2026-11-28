import os
import shlex
import sys

from motia import FlowContext, queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from steps.gvn.scripts.gvn import model_turn
from utils.event_common import check_result, get_origin_trace_id
from utils.path import (
    load_design_config,
    bebop_bin_path,
    design_output_root,
    get_buckyball_path,
    log_dir,
    simulation_dir,
)
from utils.stream_run import stream_run_logger_async

config = {
    "name": "gvn-verilator-debug",
    "description": "Cut a model into GVN slices, then check each distinct one on Verilator against BEMU",
    "flows": ["gvn"],
    "triggers": [queue("gvn.verilator.debug")],
    "enqueues": [],
}

# Set by bbdev itself; every other argument is the driver's.
own = {"config", "device", "design", "target", "_trace_id"}


@model_turn
async def handler(input_data: dict, ctx: FlowContext) -> None:
    trace_id = get_origin_trace_id(input_data, ctx)
    repo = get_buckyball_path()
    design, selection = input_data["design"], input_data["target"]
    # The slices of the debug run (the design's driver), each distinct one
    # then once on Verilator, which `bebop-verilator --build` built.
    driver = simulation_dir(repo, design, selection=selection) / "gvn/run.py"
    command = [
        "python3",
        str(driver),
        "--config",
        input_data["config"],
        "--device",
        input_data["device"],
    ]
    for key, value in input_data.items():
        if key in own:
            continue
        command.append(f"--{key}")
        if value is not True:
            command.append(str(value))
    result = await stream_run_logger_async(
        cmd=shlex.join(command),
        logger=ctx.logger,
        cwd=repo,
        stdout_prefix="gvn",
        stderr_prefix="gvn",
    )
    if result.returncode:
        await check_result(
            ctx,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "gvn", "driver": str(driver)},
            trace_id=trace_id,
        )
        return
    out = os.path.join(
        design_output_root(repo, design, selection=selection),
        "gvn",
        input_data["model"],
    )
    logs = log_dir(
        repo,
        design,
        "gvn-verilator-debug",
        input_data["model"],
        selection=selection,
    )
    result = await stream_run_logger_async(
        cmd=shlex.join(
            [
                "python3",
                os.path.join(repo, "bbdev/api/steps/gvn/scripts/verilator.py"),
                out,
                bebop_bin_path(repo, design, "verilator", selection=selection),
                logs,
                # As the regression runner: half the host, each model taking
                # its threads.
                str(
                    max(
                        1,
                        len(os.sched_getaffinity(0))
                        // load_design_config(
                            repo, design, selection=selection
                        ).sims.verilator_threads
                        // 2,
                    )
                ),
                "turn",
                "debug",
            ]
        ),
        logger=ctx.logger,
        cwd=repo,
        stdout_prefix="gvn verilator",
        stderr_prefix="gvn verilator",
    )
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "gvn-verilator-debug",
            "checks": os.path.join(logs, "verilator-debug.json"),
        },
        trace_id=trace_id,
    )
