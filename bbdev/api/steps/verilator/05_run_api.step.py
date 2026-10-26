from pathlib import Path
from utils.path import design_output_root
from utils.event_common import require_target
import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_design
from utils.path import get_buckyball_path
from utils.search_workload import search_workload

config = {
    "name": "verilator-run-api",
    "description": "trigger complete verilator workflow",
    "flows": ["verilator"],
    "triggers": [api("POST", "/verilator/run")],
    "enqueues": ["verilator.run"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        selection = require_target(body, ("device", "core"))
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    try:
        design = require_design(body)
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})
    binary = body.get("binary", "")
    if not binary:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "binary parameter is required",
            },
        )

    search_dir = str(
        Path(
            design_output_root(
                get_buckyball_path(), body["design"], selection=selection
            )
        )
        / "workloads"
    )
    if search_workload(search_dir, binary) is None:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 1,
                "error": "binary_not_found",
                "binary": binary,
                "search_dir": search_dir,
            },
        )

    data = {
        "design": design,
        "binary": binary,
        "jobs": body.get("jobs", "16"),
        "batch": body.get("batch", False),
        "coverage": body.get("coverage", False),
        "no-wave": body.get("no-wave", body.get("no_wave", False)),
        "itrace": body.get("itrace", False),
        "mtrace": body.get("mtrace", False),
        "pmctrace": body.get("pmctrace", False),
        "ctrace": body.get("ctrace", False),
        "banktrace": body.get("banktrace", False),
        "from_run_workflow": True,
    }

    await ctx.enqueue(
        {
            "topic": "verilator.run",
            "data": {**data, "_trace_id": ctx.trace_id, "target": selection},
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
