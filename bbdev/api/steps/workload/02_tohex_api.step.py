from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "workload-tohex-api",
    "description": "convert elf to hex",
    "flows": ["workload"],
    "triggers": [api("POST", "/workload/tohex")],
    "enqueues": ["workload.tohex"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}
    data = {}
    await ctx.enqueue({"topic": "workload.tohex", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
