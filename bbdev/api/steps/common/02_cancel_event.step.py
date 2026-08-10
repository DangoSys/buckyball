import asyncio

from motia import FlowContext, queue

from utils.event_common import get_origin_trace_id
from utils.process_registry import cancel_process, force_cancel_process, process_pid


config = {
    "name": "task-cancel",
    "description": "Terminate a task process group",
    "flows": ["common"],
    "triggers": [queue("common.task.cancel")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    trace_id = get_origin_trace_id(input_data, ctx)
    pid = process_pid(trace_id)
    found = cancel_process(trace_id)
    if pid is not None:
        await asyncio.sleep(5)
        force_cancel_process(trace_id, pid)
    await ctx.state.set(
        trace_id,
        "cancelled",
        {
            "status": 200,
            "body": {
                "success": False,
                "failure": False,
                "cancelled": True,
                "processing": False,
                "trace_id": trace_id,
                "process_found": found,
            },
        },
    )
