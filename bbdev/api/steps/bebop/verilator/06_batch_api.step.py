from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir

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

    try:
        chip = require_chip(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    test_type = body.get("test")
    if not test_type:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --test must be specified (elf-tests or pk-tests)"},
        )

    if test_type not in ["elf-tests", "pk-tests"]:
        return ApiResponse(
            status=400,
            body={"error": f"Invalid test type: {test_type}. Must be 'elf-tests' or 'pk-tests'"},
        )

    diff = bool(body.get("diff", False))
    rushB = bool(body.get("rushB", False))
    if diff and rushB:
        return ApiResponse(
            status=400,
            body={"error": "--diff and --rushB cannot be used together"},
        )

    vsrc_dir = rtl_dir(bbdir, chip, "verilog", body.get("vsrc_dir"))

    data = {
        "chip": chip,
        "vsrc_dir": vsrc_dir,
        "test": test_type,
        "clean-before": body.get("clean-before", body.get("clean_before", False)),
        "rushB": rushB,
        "diff": diff,
    }
    await ctx.enqueue({"topic": "bebop.verilator.batch", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
