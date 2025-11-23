import asyncio
from utils.event_common import wait_for_result


from utils.path import get_buckyball_path


config = {
    "type": "api",
    "name": "palladium Verilog",
    "description": "generate verilog code",
    "path": "/palladium/verilog",
    "method": "POST",
    "emits": ["palladium.verilog"],
    "flows": ["palladium"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()
    body = req.get("body") or {}

    # Get config name, must be provided
    config_name = body.get("config")
    if not config_name or config_name == "None":
        return {
            "status": "error",
            "message": "Configuration name is required. Please specify --config_name parameter.",
            "example": './bbdev palladium --verilog "--config_name sims.palladium.BuckyballToyP2EConfig"',
        }

    data = {
        "config": config_name,
        "balltype": body.get("balltype"),
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
    }
    await context.emit({"topic": "palladium.verilog", "data": data})

    # ==================================================================================
    #  Wait for simulation result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
