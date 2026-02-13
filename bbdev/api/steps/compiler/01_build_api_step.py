import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Build Compiler",
    "description": "build bitstream",
    "path": "/compiler/build",
    "method": "POST",
    "emits": ["compiler.build"],
    "flows": ["compiler"],
}


async def handler(req, context):
    body = req.get("body") or {}
    await context.emit({"topic": "compiler.build", "data": body})

    # ==================================================================================
    #  Wait for build result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
