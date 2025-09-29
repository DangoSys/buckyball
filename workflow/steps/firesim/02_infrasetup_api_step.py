import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Firesim Infrasetup",
    "description": "infrasetup",
    "path": "/firesim/infrasetup",
    "method": "POST",
    "emits": ["firesim.infrasetup"],
    "flows": ["firesim"],
}


async def handler(req, context):
    body = req.get("body") or {}
    data = {"jobs": body.get("jobs", 16)}
    await context.emit({"topic": "firesim.infrasetup", "data": data})

    # ==================================================================================
    #  等待仿真结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
