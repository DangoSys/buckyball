from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design

config = {
    "name": "bebop-verilator-batch-api",
    "description": "Run bebop verilator batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/batch")],
    "enqueues": ["bebop.verilator.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device",))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    diff = bool(body.get("diff", False))
    test_type = body.get("test")
    if not test_type:
        return ApiResponse(
            status=400,
            body={
                "error": "Missing required parameter: --test must be specified (bare-tests or linux-tests)"
            },
        )
    if test_type not in ("bare-tests", "linux-tests"):
        return ApiResponse(
            status=400,
            body={
                "error": f"Invalid test type: {test_type}. Must be 'bare-tests' or 'linux-tests'"
            },
        )
    data = {
        "design": design,
        "test": test_type,
        "jobs": body.get("jobs", 1),
        "diff": diff,
    }
    await ctx.enqueue(
        {
            "topic": "bebop.verilator.batch",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
