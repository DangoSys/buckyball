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
    "name": "running sardine",
    "description": "running sardine",
    "path": "/sardine/run",
    "method": "POST",
    "emits": ["sardine.run"],
    "flows": ["sardine"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()

    body = req.get("body") or {}

    data = {"workload": body.get("workload", "")}

    sardine_dir = f"{bbdir}/bb-tests/sardine"

    await context.emit({"topic": "sardine.run", "data": data})

    # ==================================================================================
    # 等待执行结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
