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
    memory = body.get("guest-memory-mib", 512)
    if isinstance(memory, bool) or not str(memory).isdigit() or not 1 <= int(memory) <= 16384:
        return ApiResponse(status=400, body={"error": "guest-memory-mib must be an integer in 1..16384 (P2E DDR capacity)"})
    storage = body.get("model-storage", "initramfs")
    if storage not in ("initramfs", "pmem"):
        return ApiResponse(status=400, body={"error": "model-storage must be initramfs or pmem"})
    if storage == "pmem" and (not body.get("model") or int(memory) >= 16384):
        return ApiResponse(status=400, body={"error": "pmem requires --model and guest-memory-mib < 16384"})
    await ctx.enqueue({"topic": "kernel.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
