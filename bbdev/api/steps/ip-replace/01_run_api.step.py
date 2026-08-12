from motia import ApiRequest, ApiResponse, FlowContext, api


def req_arg(body: dict, name: str):
    return body.get(name) or body.get(name.replace("_", "-"))


config = {
    "name": "ip-replace-api",
    "description": "prepare top-scoped synthesis RTL and SRAM metadata",
    "flows": ["ip-replace"],
    "triggers": [api("POST", "/ip/replace/run")],
    "enqueues": ["ip-replace.run"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    source_list = req_arg(body, "source_list")
    if not isinstance(source_list, str) or not source_list:
        return ApiResponse(status=400, body={"error": "source_list is required"})
    output_dir = req_arg(body, "output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return ApiResponse(status=400, body={"error": "output_dir is required"})

    data = {
        "source_list": source_list,
        "ip_replace_output_dir": output_dir,
        "top": req_arg(body, "top"),
        "consumer": req_arg(body, "consumer") or "generic",
    }
    await ctx.enqueue({"topic": "ip-replace.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
