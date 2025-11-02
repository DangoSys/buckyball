import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Marshal Build",
    "description": "build marshal",
    "path": "/marshal/build",
    "method": "POST",
    "emits": ["marshal.build"],
    "flows": ["marshal"],
}


async def handler(req, context):
    body = req.get("body") or {}
    await context.emit({"topic": "marshal.build", "data": body})
    # ==================================================================================
    #  Wait for result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
