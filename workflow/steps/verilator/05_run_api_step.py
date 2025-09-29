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
        "jobs": body.get("jobs", "16"),
        "batch": body.get("batch", False),
        "from_run_workflow": True,
    }

    await context.emit({"topic": "verilator.run", "data": config})

    # ==================================================================================
    #  等待仿真结果
    #
    #  期望返回结果是：
    #  {
    #    "status": 200/400/500,
    #    "body": {
    #      "success": true/false,
    #      "failure": true/false,
    #      "processing": true/false,
    #      "return_code": 0,
    #      其余字段
    #    }
    #  }
    #
    #  由于Motia框架会把数据包装在data字段中，所以需要解包
    #       if isinstance(result, dict) and 'data' in result:
    #          return result['data']
    #       return result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
