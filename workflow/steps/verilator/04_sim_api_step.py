import os
import sys
import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Verilator Sim",
    "description": "run verilator simulation",
    "path": "/verilator/sim",
    "method": "POST",
    "emits": ["verilator.sim"],
    "flows": ["verilator"],
}


async def handler(req, context):
    body = req.get("body") or {}
    binary = body.get("binary", "")
    batch = body.get("batch", False)
    if not binary:
        return {
            "status": 400,
            "body": {
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is required",
            },
        }

    await context.emit({"topic": "verilator.sim", "data": {**body, "task": "sim"}})
    # ==================================================================================
    #  Wait for simulation result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
