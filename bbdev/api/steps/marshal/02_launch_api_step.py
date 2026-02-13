import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Marshal Launch",
    "description": "launch marshal",
    "path": "/marshal/launch",
    "method": "POST",
    "emits": ["marshal.launch"],
    "flows": ["marshal"],
}


async def handler(req, context):
    body = req.get("body") or {}
    await context.emit({"topic": "marshal.launch", "data": body})

    # ==================================================================================
    #  Wait for result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
