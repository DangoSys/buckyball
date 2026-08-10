from motia import ApiRequest, ApiResponse, FlowContext, api


config = {
    "name": "task-cancel-api",
    "description": "Cancel a queued or running bbdev task by trace_id",
    "flows": ["common"],
    "triggers": [api("POST", "/task/{trace_id}/cancel")],
    "enqueues": ["common.task.cancel"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    trace_id = request.path_params.get("trace_id", "")
    if not trace_id:
        return ApiResponse(status=400, body={"error": "trace_id is required"})

    for key in ("success", "failure", "cancelled"):
        if await ctx.state.get(trace_id, key):
            return ApiResponse(
                status=409,
                body={"error": "task is already terminal", "trace_id": trace_id},
            )

    await ctx.enqueue({
        "topic": "common.task.cancel",
        "data": {"_trace_id": trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": trace_id, "status": "cancelling"})
