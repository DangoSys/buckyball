import asyncio
from utils.event_common import wait_for_result


from utils.path import get_buckyball_path


config = {
    "type": "api",
    "name": "Yosys Synth",
    "description": "run yosys synthesis for area estimation",
    "path": "/yosys/synth",
    "method": "POST",
    "emits": ["yosys.synth"],
    "flows": ["yosys"],
}


async def handler(req, context):
    bbdir = get_buckyball_path()
    body = req.get("body") or {}

    data = {
        "balltype": body.get("balltype"),
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
        "top": body.get("top"),
    }
    await context.emit({"topic": "yosys.synth", "data": data})

    # ==================================================================================
    #  Wait for synthesis result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
