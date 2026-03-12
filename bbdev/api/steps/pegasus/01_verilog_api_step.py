import asyncio
from utils.event_common import wait_for_result
from utils.path import get_buckyball_path


config = {
    "type": "api",
    "name": "Pegasus Verilog",
    "description": "Generate SystemVerilog for Pegasus FPGA (PegasusHarness + ChipTop)",
    "path": "/pegasus/verilog",
    "method": "POST",
    "emits": ["pegasus.verilog"],
    "flows": ["pegasus"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()
    body = req.get("body") or {}

    # Default config for Pegasus; allow override
    config_name = body.get("config", "sims.pegasus.PegasusConfig")

    data = {
        "config": config_name,
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/pegasus/"),
    }

    await context.emit({"topic": "pegasus.verilog", "data": data})

    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
