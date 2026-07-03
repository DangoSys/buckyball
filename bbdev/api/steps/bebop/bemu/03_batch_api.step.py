import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.bemu import validate_bemu_chip
from utils.path import get_buckyball_path

config = {
    "name": "bebop-bemu-batch-api",
    "description": "Run bebop bemu nextest batch regression",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/bemu/batch")],
    "enqueues": ["bebop.bemu.batch"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}

    chip = body.get("chip")
    try:
        validate_bemu_chip(chip, get_buckyball_path())
    except ValueError as e:
        return ApiResponse(
            status=400,
            body={"error": str(e)}
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

    await ctx.enqueue({
        "topic": "bebop.bemu.batch",
        "data": {"chip": chip, "test": test_type, "_trace_id": ctx.trace_id}
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
