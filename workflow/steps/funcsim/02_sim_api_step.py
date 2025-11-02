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
    "name": "run functional simulator",
    "description": "run functional simulator",
    "path": "/funcsim/sim",
    "method": "POST",
    "emits": ["funcsim.sim"],
    "flows": ["funcsim"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()
    body = req.get("body") or {}
    data = {
        "funcsim": body.get("funcsim", ""),
        "binary": body.get("binary", ""),
        "ext": body.get("ext", ""),
    }
    if not data.get("binary"):
        context.logger.error(
            "binary parameter is missing in event data", {"data": data}
        )
        return
    if not data.get("ext"):
        context.logger.error(
            "ext parameter is missing in event data",
            {"data": data, "ext": data.get("ext")},
        )
        return

    await context.emit({"topic": "funcsim.sim", "data": data})

    # ==================================================================================
    #  Wait for simulation result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
