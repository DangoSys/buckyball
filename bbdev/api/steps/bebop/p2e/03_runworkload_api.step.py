from utils.event_common import require_product, require_target

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
    allowed = {
        "config",
        "device",
        "hart",
        "model",
        "bench",
        "test",
        "os",
        "bitstream",
        "fpga-location",
        "fpga_location",
        "multi-fpga",
        "wave",
        "wave-start",
        "diff",
        "ddr-image",
        "model-image",
        "model-offset",
        "itrace",
        "mtrace",
        "pmctrace",
        "ctrace",
        "banktrace",
    }
    if unknown := body.keys() - allowed:
        return ApiResponse(
            status=400,
            body={"error": f"unknown parameter(s): {', '.join(sorted(unknown))}"},
        )
    try:
        selection = require_target(body, ("device", "hart"))
        require_product(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    if not body.get("bitstream"):
        return ApiResponse(status=400, body={"error": "bitstream is required"})
    if body.get("ddr-image") is not None and not body.get("diff", False):
        return ApiResponse(status=400, body={"error": "ddr-image requires diff"})
    await ctx.enqueue(
        {
            "topic": "bebop.p2e.runworkload",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
