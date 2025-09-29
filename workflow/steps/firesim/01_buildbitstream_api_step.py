import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Firesim Buildbitstream",
    "description": "build bitstream",
    "path": "/firesim/buildbitstream",
    "method": "POST",
    "emits": ["firesim.buildbitstream"],
    "flows": ["firesim"],
}


async def handler(req, context):
    body = req.get("body") or {}
    await context.emit({"topic": "firesim.buildbitstream", "data": body})

    # ==================================================================================
    #  等待仿真结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
