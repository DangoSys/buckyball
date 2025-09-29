import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Verilator Clean",
    "description": "clean build directory",
    "path": "/verilator/clean",
    "method": "POST",
    "emits": ["verilator.clean"],
    "flows": ["verilator"],
}


async def handler(req, context):
    body = req.get("body") or {}
    await context.emit({"topic": "verilator.clean", "data": {**body, "task": "clean"}})

    # ==================================================================================
    #  等待仿真结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
