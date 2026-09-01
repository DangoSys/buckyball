from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip


def req_arg(body: dict, name: str):
    return body.get(name) or body.get(name.replace("_", "-"))


config = {
    "name": "ip-generate-api",
    "description": "generate SRAM macros from elaborator mems.conf via MacroCompiler",
    "flows": ["ip", "dc", "yosys"],
    "triggers": [api("POST", "/ip/generate")],
    "enqueues": ["ip.generate"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {
        "chip": chip,
        "consumer": req_arg(body, "consumer") or "dc",
        "top": req_arg(body, "top") or "DigitalTop",
    }
    if data["consumer"] not in ("dc", "yosys"):
        return ApiResponse(status=400, body={"error": "consumer must be dc or yosys"})
    output_dir = req_arg(body, "output_dir")
    if output_dir is not None:
        if not isinstance(output_dir, str) or not output_dir:
            return ApiResponse(status=400, body={"error": "output_dir must be a non-empty string"})
        data["output_dir"] = output_dir
    next_topic = req_arg(body, "next_topic")
    if next_topic is not None:
        if not isinstance(next_topic, str) or not next_topic:
            return ApiResponse(status=400, body={"error": "next_topic must be a non-empty string"})
        data["next_topic"] = next_topic
    await ctx.enqueue({"topic": "ip.generate", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
