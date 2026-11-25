from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bench-tool-api",
    "description": "measure the simulators of a design into a bbbench snapshot",
    "flows": ["bench"],
    "triggers": [api("POST", "/bench/tool/:tool")],
    "enqueues": ["bench.tool"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    tool = request.path_params["tool"]
    # The flags each tool takes, at most one: none is its fastest.
    flags = {
        "bemu": [set(), {"all-trace"}],
        "verilator": [set(), {"all-trace"}, {"diff"}],
        "p2e": [set(), {"all-trace"}, {"diff"}],
    }
    if (
        tool not in flags
        or "device" not in body
        or set(body) - {"device"} not in flags[tool]
        or any(body[f] is not True for f in set(body) - {"device"})
    ):
        return ApiResponse(
            status=400,
            body={
                "error": (
                    'bench-tool --bemu takes "--device (all | <device>) [--all-trace]",'
                    ' --verilator or --p2e "--device (all | <device>) [--all-trace | --diff]"'
                )
            },
        )
    await ctx.enqueue(
        {
            "topic": "bench.tool",
            "data": {
                "tool": tool,
                "device": body["device"],
                "all-trace": "all-trace" in body,
                "diff": "diff" in body,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
