import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "UVM Build",
    "description": "build uvm executable",
    "path": "/uvm/build",
    "method": "POST",
    "emits": ["uvm.build"],
    "flows": ["uvm"],
}


async def handler(req, context):
    body = req.get("body") or {}
    data = {"jobs": body.get("jobs", 16)}
    await context.emit({"topic": "uvm.build", "data": data})

    # ==================================================================================
    #  Wait for build result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
