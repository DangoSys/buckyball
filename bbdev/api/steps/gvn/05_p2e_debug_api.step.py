import re

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design, require_target

config = {
    "name": "gvn-p2e-debug-api",
    "description": "Check a model's GVN slices on FPGA against BEMU",
    "flows": ["gvn"],
    "triggers": [api("POST", "/gvn/p2e/debug")],
    "enqueues": ["gvn.p2e.debug"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        if set(body) - {"device", "model", "config", "bitstream", "fpga-location"}:
            raise ValueError("Unexpected FPGA GVN argument")
        if not isinstance(body.get("model"), str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", body["model"]
        ):
            raise ValueError("--model requires a model name")
        if "fpga-location" in body and not re.fullmatch(
            r"[0-9]+\.[A-E]", str(body["fpga-location"])
        ):
            raise ValueError("--fpga-location requires a slot such as 0.A")
        if "bitstream" in body and not isinstance(body["bitstream"], str):
            raise ValueError("--bitstream requires a path")
        selection = require_target(body, ("device",))
        design = require_design(body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "gvn.p2e.debug",
            "data": {
                **body,
                "design": design,
                "target": selection,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
