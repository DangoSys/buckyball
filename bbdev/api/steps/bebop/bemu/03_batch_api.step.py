from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-bemu-batch-api",
    "description": "Run bebop bemu nextest batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/batch")],
    "enqueues": ["bebop.bemu.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    test_type = body.get("test")
    if not test_type:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --test must be specified (elf-tests or pk-tests)"}
        )

    if test_type not in ["elf-tests", "pk-tests"]:
        return ApiResponse(
            status=400,
            body={"error": f"Invalid test type: {test_type}. Must be 'elf-tests' or 'pk-tests'"}
        )

    await ctx.enqueue({
        "topic": "bebop.bemu.batch",
        "data": {"test": test_type, "_trace_id": ctx.trace_id}
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
