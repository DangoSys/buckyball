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
        error_msg = "Configuration name is required. Please specify --config parameter."
        context.logger.error(error_msg, {"config_name": config_name})
        return {
            "status": 400,
            "body": {
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "Configuration name is required. Please specify --config parameter.",
            },
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
