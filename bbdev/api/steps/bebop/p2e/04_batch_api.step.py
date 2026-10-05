from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-p2e-batch-api",
    "description": "Run bebop p2e batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/batch")],
    "enqueues": ["bebop.p2e.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    bitstream = body.get("bitstream", "")
    if not bitstream:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "--bitstream parameter is required",
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
            body={"error": "Missing required parameter: --test must be specified (bare-tests or linux-tests)"}
        )

    if test_type not in ["bare-tests", "linux-tests"]:
        return ApiResponse(
            status=400,
            body={"error": f"Invalid test type: {test_type}. Must be 'bare-tests' or 'linux-tests'"}
        )

    fpga_location = body.get("fpga-location", "0.A")
    if not isinstance(fpga_location, str) or not fpga_location:
        return ApiResponse(status=400, body={"error": "--fpga-location must be a location such as 1.A"})

    diff = bool(body.get("diff", False))
    data = {
        "chip": chip,
        "bitstream": bitstream,
        "test": test_type,
        "diff": diff,
        "fpga-location": fpga_location,
    }
    await ctx.enqueue({"topic": "bebop.p2e.batch", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
