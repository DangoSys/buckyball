import asyncio
from utils.event_common import wait_for_result

config = {
    "type": "api",
    "name": "Verilator Complete Workflow",
    "description": "trigger complete verilator workflow",
    "path": "/verilator/run",
    "method": "POST",
    "emits": ["verilator.run"],
    "flows": ["verilator"],
}


async def handler(req, context):
    body = req.get("body") or {}

    config = {
        "binary": body.get("binary", ""),
        "config": body.get("config", "sims.verilator.BuckyballToyVerilatorConfig"),
        "jobs": body.get("jobs", "16"),
        "batch": body.get("batch", False),
        "from_run_workflow": True,
    }

    await context.emit({"topic": "verilator.run", "data": config})

    # ==================================================================================
    #  Wait for simulation result
    #
    #  Expected return result format:
    #  {
    #    "status": 200/400/500,
    #    "body": {
    #      "success": true/false,
    #      "failure": true/false,
    #      "processing": true/false,
    #      "return_code": 0,
    #      other fields
    #    }
    #  }
    #
    #  Since the Motia framework wraps data in the data field, it needs to be unpacked
    #       if isinstance(result, dict) and 'data' in result:
    #          return result['data']
    #       return result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
