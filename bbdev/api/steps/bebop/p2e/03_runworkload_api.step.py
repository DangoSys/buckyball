from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bebop-p2e-runworkload-api",
    "description": "Run workload on FPGA via bebop p2e CLI",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/runworkload")],
    "enqueues": ["bebop.p2e.runworkload"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    image = body.get("image", "")
    bitstream = body.get("bitstream", "")
    if not image or not bitstream:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "image and bitstream parameters are required",
            },
        )

    await ctx.enqueue({
        "topic": "bebop.p2e.runworkload",
        "data": {**body, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
