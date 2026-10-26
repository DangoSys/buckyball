from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design
from utils.path import get_buckyball_path

config = {
    "name": "bebop-p2e-verilog-api",
    "description": "Generate SystemVerilog for P2E DDR4 backdoor",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/verilog")],
    "enqueues": ["bebop.p2e.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    data = {"design": design, "diff": bool(body.get("diff", False))}
    if body.get("output-dir"):
        data["output_dir"] = body["output-dir"]
    await ctx.enqueue(
        {
            "topic": "bebop.p2e.verilog",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
