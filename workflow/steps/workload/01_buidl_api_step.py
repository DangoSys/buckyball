import subprocess
import sys
import os
import asyncio
from utils.event_common import wait_for_result

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
    "type": "api",
    "name": "build workload",
    "description": "build workload",
    "path": "/workload/build",
    "method": "POST",
    "emits": ["workload.build"],
    "flows": ["workload"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()
    body = req.get("body") or {}
    data = {"workload": body.get("workload", "")}
    workload_dir = f"{bbdir}/bb-tests/workload"
    await context.emit({"topic": "workload.build", "data": data})

    # ==================================================================================
    #  等待仿真结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
