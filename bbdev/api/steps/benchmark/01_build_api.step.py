from pathlib import Path

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.event_common import require_target
from utils.path import get_buckyball_path

config = {
    "name": "bench-build-api",
    "description": "Build the benchmarks of a simulation",
    "flows": ["workload"],
    "triggers": [api("POST", "/bench/build")],
    "enqueues": ["workload.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    # The benchmarks: the suites under bb-tests/workloads/src/benchmark.
    benches = sorted(
        p.name
        for p in (
            Path(get_buckyball_path()) / "bb-tests/workloads/src/benchmark"
        ).iterdir()
        if p.is_dir()
    )
    try:
        selection = require_target(body, ("device", "hart"))
        if body.get("bench") not in ("all", *benches):
            raise ValueError(f"--bench (all | {' | '.join(benches)}) is required")
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue(
        {
            "topic": "workload.build",
            "data": {
                "design": body["design"],
                "bench": body["bench"],
                "target": selection,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
