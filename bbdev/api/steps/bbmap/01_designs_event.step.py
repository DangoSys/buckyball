import os
import subprocess
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path

config = {
    "name": "bbmap-designs",
    "description": "map every design: what it is for, its models, configs and composition",
    "flows": ["bbmap"],
    "triggers": [queue("bbmap.designs")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin = get_origin_trace_id(input_data, ctx)
    repo = Path(get_buckyball_path())
    out = repo / "thirdparty/bbbench/save/bbmap.json"
    # design.pb is read in its own process: generated protobuf modules must
    # not share a long-lived server's descriptor pool.
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "scripts/designs.py"),
            str(repo),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    ctx.logger.info(result.stdout)
    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "designs",
            "map": str(out.relative_to(repo)),
            "error": result.stderr[-2000:],
        },
        trace_id=origin,
    )
