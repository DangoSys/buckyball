from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "kernel-build-api",
    "description": "build RISC-V kernel + rootfs for Pegasus",
    "flows": ["kernel"],
    "triggers": [api("POST", "/kernel/build")],
    "enqueues": ["kernel.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    await ctx.enqueue({"topic": "kernel.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
