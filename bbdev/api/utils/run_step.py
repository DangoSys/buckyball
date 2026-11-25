import asyncio
import time
import uuid


async def run(
    ctx, topic: str, data: dict, limit: float | None = None
) -> tuple[str, dict]:
    """Run one step as its own task and wait for its terminal state; past
    `limit` seconds it is cancelled."""
    trace = str(uuid.uuid4())
    await ctx.enqueue({"topic": topic, "data": {**data, "_trace_id": trace}})
    start = time.monotonic()
    while True:
        if limit is not None and time.monotonic() - start > limit:
            await ctx.enqueue(
                {"topic": "common.task.cancel", "data": {"_trace_id": trace}}
            )
            return "timeout", {
                "error": f"{topic} ran past {limit // 60:.0f} minutes and was cancelled"
            }
        for state in ("success", "failure", "cancelled"):
            try:
                value = await ctx.state.get(trace, state)
            except TimeoutError:
                # The state service of a loaded host answers late; the next
                # poll asks again.
                break
            if value:
                return state, value.get("body", value)
        await asyncio.sleep(10)
