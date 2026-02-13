import asyncio
from utils.event_common import wait_for_result


from utils.path import get_buckyball_path


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
    bbdir = get_buckyball_path()
    body = req.get("body") or {}

    # Get config name, must be provided
    config_name = body.get("config")
    if not config_name or config_name == "None":
        return {
            "status": "error",
            "message": "Configuration name is required. Please specify --config parameter.",
            "example": 'bbdev verilator --verilog "--config sims.verilator.BuckyballToyVerilatorConfig"',
        }

    data = {
        "config": config_name,
        "balltype": body.get("balltype"),
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
    }
    await context.emit({"topic": "verilator.verilog", "data": data})

    # ==================================================================================
    #  Wait for simulation result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
