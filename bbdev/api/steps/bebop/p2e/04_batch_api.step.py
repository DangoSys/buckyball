from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-p2e-batch-api",
    "description": "Run bebop p2e nextest batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/batch")],
    "enqueues": ["bebop.p2e.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    bitstream = body.get("bitstream", "")
    build_dir = body.get("build-dir") or body.get("build_dir", "")
    if not bitstream or not build_dir:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "--bitstream and --build-dir parameters are required",
            },
        )

    chip = body.get("chip")
    if not chip:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --chip must be specified"}
        )

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

    data = {
        "chip": chip,
        "bitstream": bitstream,
        "build_dir": build_dir,
        "test": test_type,
    }
    await ctx.enqueue({"topic": "bebop.p2e.batch", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
