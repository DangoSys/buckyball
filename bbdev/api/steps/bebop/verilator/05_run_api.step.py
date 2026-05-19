from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir

config = {
    "name": "bebop-verilator-run-api",
    "description": "Run complete bebop verilator workflow: clean → verilog → build → sim",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/verilator/run")],
    "enqueues": ["bebop.verilator.run"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    config_name = body.get("config")
    if not config_name:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --config must be specified"}
        )

    build_dir = get_verilator_build_dir(bbdir, config_name, body.get("output_dir"))

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

    data = {
        "config": config_name,
        "output_dir": build_dir,
        "binary": binary,
        "balltype": body.get("balltype"),
        "itrace": body.get("itrace", False),
        "mtrace": body.get("mtrace", False),
        "pmctrace": body.get("pmctrace", False),
        "ctrace": body.get("ctrace", False),
        "banktrace": body.get("banktrace", False),
        "log_dir": body.get("log_dir"),
        "fst_dir": body.get("fst_dir"),
        "from_run_workflow": True,
    }
    await ctx.enqueue({"topic": "bebop.verilator.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
