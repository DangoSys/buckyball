import os
import re
import shlex
import sys

from motia import FlowContext, queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path, simulation_dir
from utils.stream_run import stream_run_logger_async

config = {
    "name": "gvn-bemu-debug",
    "description": "Run the design's GVN driver on one model",
    "flows": ["gvn"],
    "triggers": [queue("gvn.bemu.debug")],
    "enqueues": [],
}

# Set by bbdev itself; every other argument is the driver's.
own = {"config", "device", "design", "target", "_trace_id"}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    trace_id = get_origin_trace_id(input_data, ctx)
    repo = get_buckyball_path()
    # A design's driver is gvn/run.py of its simulation entry: designs differ
    # in how their harts and threads are cut, so each owns its driver.
    driver = (
        simulation_dir(repo, input_data["design"], selection=input_data["target"])
        / "gvn/run.py"
    )
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
    fields = {"task": "gvn", "driver": str(driver)}
    captured = re.search(r"^summary: .+; logs and ELFs: (.+)$", result.stdout, re.M)
    if result.returncode == 0 and captured:
        fields["manifest"] = os.path.join(captured[1], "instances.jsonl")
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=fields,
        trace_id=trace_id,
    )
