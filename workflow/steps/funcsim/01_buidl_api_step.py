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
    "name": "build functional simulator",
    "description": "build functional simulator",
    "path": "/funcsim/build",
    "method": "POST",
    "emits": ["funcsim.build"],
    "flows": ["funcsim"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()
    body = req.get("body") or {}
    funcsim_dir = f"{bbdir}/sims/func-sim"
    data = {"funcsim": body.get("funcsim", "")}
    funcsim_dir = f"{bbdir}/sims/func-sim"
    await context.emit({"topic": "funcsim.build", "data": data})

    # ==================================================================================
    #  Wait for simulation result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
