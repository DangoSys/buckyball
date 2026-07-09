from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-verilator-batch-api",
    "description": "Run bebop verilator nextest batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/batch")],
    "enqueues": ["bebop.verilator.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    arch_config = body.get("config")
    if not isinstance(arch_config, str) or not arch_config or arch_config == "None":
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"}
        )

    test_type = body.get("test")
    if not test_type:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --test must be specified (elf-tests or pk-tests)"}
        )

    if test_type not in ["elf-tests", "pk-tests"]:
        return ApiResponse(
            status=400,
            body={"error": f"Invalid test type: {test_type}. Must be 'elf-tests' or 'pk-tests'"}
        )

    vsrc_dir = get_verilator_build_dir(bbdir, arch_config, body.get("vsrc_dir"))

    data = {
        "config": arch_config,
        "vsrc_dir": vsrc_dir,
        "test": test_type,
        "clean-before": body.get("clean-before", body.get("clean_before", False)),
    }
    await ctx.enqueue({"topic": "bebop.verilator.batch", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
