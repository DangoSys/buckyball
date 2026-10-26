from utils.event_common import require_target

from motia import ApiRequest, ApiResponse, FlowContext, api
from steps.bebop.p2e.scripts.runtime_case import validate_runtime_reuse

config = {
    "name": "bebop-p2e-runworkload-api",
    "description": "Run workload on FPGA via bebop p2e CLI",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/runworkload")],
    "enqueues": ["bebop.p2e.runworkload"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    if not body.get("image") or not body.get("bitstream"):
        return ApiResponse(
            status=400, body={"error": "image and bitstream are required"}
        )
    bitstream = body["bitstream"]
    if body.get("reuse-runtime", False):
        try:
            validate_runtime_reuse(
                bitstream,
                bool(body.get("diff", False)),
                bool(body.get("itrace", False)),
                bool(body.get("mtrace", False)),
            )
        except (ValueError, OSError) as error:
            return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "bebop.p2e.runworkload",
            "data": {**body, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
