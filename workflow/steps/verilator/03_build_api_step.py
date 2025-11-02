import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Verilator Build",
    "description": "build verilator executable",
    "path": "/verilator/build",
    "method": "POST",
    "emits": ["verilator.build"],
    "flows": ["verilator"],
}


async def handler(req, context):
    body = req.get("body") or {}
    data = {"jobs": body.get("jobs", 16)}
    await context.emit({"topic": "verilator.build", "data": data})

    # ==================================================================================
    #  Wait for simulation result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
