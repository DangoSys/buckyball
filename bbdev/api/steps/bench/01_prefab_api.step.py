from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bench-prefab-api",
    "description": "evaluate prefab components into a bbbench snapshot",
    "flows": ["bench"],
    "triggers": [api("POST", "/bench/prefab/:evaluation")],
    "enqueues": ["bench.prefab"],
}

# What each kind of component is evaluated by.
evaluations = {
    "ball": ("area", "uvm", "performance", "power"),
    "thread": ("test", "area", "power"),
    "hart": ("test", "area", "power"),
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    evaluation = request.path_params["evaluation"]
    kinds = [kind for kind in evaluations if kind in body]
    if len(kinds) != 1 or set(body) != set(kinds):
        return ApiResponse(
            status=400,
            body={
                "error": "Specify exactly one of --ball (all | <ball>),"
                " --thread (all | <thread>), --hart (all | <hart>)"
            },
        )
    (kind,) = kinds
    if not isinstance(body[kind], str) or not body[kind]:
        return ApiResponse(
            status=400, body={"error": f"--{kind} (all | <{kind}>) is required"}
        )
    if evaluation not in evaluations[kind]:
        return ApiResponse(
            status=400,
            body={
                "error": f"A {kind} is evaluated by "
                + ", ".join(f"--{k}" for k in evaluations[kind])
            },
        )
    await ctx.enqueue(
        {
            "topic": "bench.prefab",
            "data": {kind: body[kind], evaluation: True, "_trace_id": ctx.trace_id},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
