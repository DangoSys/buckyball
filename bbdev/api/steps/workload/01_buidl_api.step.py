from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "workload-build-api",
    "description": "build workload",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/build")],
    "enqueues": ["workload.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}
    data = {"workload": body.get("workload", "")}
    await ctx.enqueue({"topic": "workload.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
