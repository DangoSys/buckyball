from utils.event_common import require_target
import os
import sys
from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
from utils.event_common import require_design

config = {
    "name": "bebop-bemu-batch-api",
    "description": "Run bebop bemu batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/batch")],
    "enqueues": ["bebop.bemu.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device",))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    design = body.get("design")
    try:
        require_design({"design": design})
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
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
    await ctx.enqueue(
        {
            "topic": "bebop.bemu.batch",
            "data": {
                "design": design,
                "test": test_type,
                "jobs": body.get("jobs", 1),
                "_trace_id": ctx.trace_id,
                "target": selection,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
