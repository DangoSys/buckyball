from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "sardine-run-api",
    "description": "running sardine",
    "flows": ["sardine"],
    "triggers": [api("POST", "/sardine/run")],
    "enqueues": ["sardine.run"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()

    body = request.body or {}

    data = {"workload": body.get("workload", "")}

    await ctx.enqueue({"topic": "sardine.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
