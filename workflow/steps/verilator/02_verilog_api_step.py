import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Verilator Verilog",
    "description": "generate verilog code",
    "path": "/verilator/verilog",
    "method": "POST",
    "emits": ["verilator.verilog"],
    "flows": ["verilator"],
}


async def handler(req, context):
    body = req.get("body") or {}
    data = {"balltype": body.get("balltype")}
    await context.emit({"topic": "verilator.verilog", "data": data})

    # ==================================================================================
    #  等待仿真结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
