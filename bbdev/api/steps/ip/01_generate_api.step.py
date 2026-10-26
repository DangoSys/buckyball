from utils.event_common import require_target
from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_design


def req_arg(body: dict, name: str):
    return body.get(name) or body.get(name.replace("_", "-"))


config = {
    "name": "ip-generate-api",
    "description": "generate SRAM macro wrappers from elaborator mems.conf via the process sram function",
    "flows": ["ip", "dc", "yosys"],
    "triggers": [api("POST", "/ip/generate")],
    "enqueues": ["ip.generate"],
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

    data = {
        "design": design,
        "consumer": req_arg(body, "consumer") or "dc",
        "top": req_arg(body, "top"),
    }
    if data["consumer"] not in ("dc", "yosys"):
        return ApiResponse(status=400, body={"error": "consumer must be dc or yosys"})
    output_dir = req_arg(body, "output_dir")
    if output_dir is not None:
        if not isinstance(output_dir, str) or not output_dir:
            return ApiResponse(
                status=400, body={"error": "output_dir must be a non-empty string"}
            )
        data["output_dir"] = output_dir
    next_topic = req_arg(body, "next_topic")
    if next_topic is not None:
        if not isinstance(next_topic, str) or not next_topic:
            return ApiResponse(
                status=400, body={"error": "next_topic must be a non-empty string"}
            )
        data["next_topic"] = next_topic
    await ctx.enqueue(
        {
            "topic": "ip.generate",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
