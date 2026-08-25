import os
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path, rtl_dir


config = {
    "name": "vcs-clean",
    "description": "remove simulator-specific VCS artifacts",
    "flows": ["vcs"],
    "triggers": [queue("vcs.clean")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = rtl_dir(bbdir, input_data.get("config", "verilog"), input_data.get("output_dir"))
    artifact_dir = os.path.join(build_dir, "vcs")
    if os.path.isdir(artifact_dir):
        shutil.rmtree(artifact_dir)
    await check_result(ctx, 0, extra_fields={"task": "clean", "artifact_dir": artifact_dir}, trace_id=origin_tid)
