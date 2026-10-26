from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api
from utils.event_common import require_design

config = {
    "name": "bebop-verilator-run-api",
    "description": "Run complete bebop verilator workflow: clean → verilog → build → sim",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/run")],
    "enqueues": ["bebop.verilator.run.clean"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    binary = body.get("binary", "")
    if not binary:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is required",
            },
        )
    diff = body.get("diff", False)
    data = {
        "design": design,
        "binary": binary,
        "itrace": body.get("itrace", False),
        "mtrace": body.get("mtrace", False),
        "pmctrace": body.get("pmctrace", False),
        "ctrace": body.get("ctrace", False),
        "banktrace": body.get("banktrace", False),
        "diff": diff,
        "no-wave": body.get("no-wave", body.get("no_wave", False)),
        "jobs": body.get("jobs", 16),
        "from_run_workflow": True,
    }
    await ctx.enqueue(
        {
            "topic": "bebop.verilator.run.clean",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
