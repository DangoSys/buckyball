from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-verilator-build-api",
    "description": "Build bebop verilator binary",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/build")],
    "enqueues": ["bebop.verilator.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    unsupported = [name for name in ("vsrc_dir", "output_dir") if name in body]
    if unsupported:
        return ApiResponse(
            status=400,
            body={"error": f"Unsupported parameter(s): {', '.join(unsupported)}. Use --vsrc-dir/--output-dir."},
        )

    arch_config = body.get("config")
    if not isinstance(arch_config, str) or not arch_config or arch_config == "None":
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"}
        )

    data = {
        "config": arch_config,
        "jobs": body.get("jobs", 16),
        "diff": bool(body.get("diff", False)),
        "vsrc_dir": get_verilator_build_dir(
            bbdir,
            arch_config,
            body.get("vsrc-dir") or body.get("output-dir"),
        ),
    }
    await ctx.enqueue({"topic": "bebop.verilator.build", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
