from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "dc-run-api",
    "description": "run Design Compiler synthesis script",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/run")],
    "enqueues": ["dc.run"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = request.body or {}

    srcdir = body.get("srcdir")
    if not srcdir:
        return ApiResponse(
            status=400,
            body={
                "status": "error",
                "message": "Missing required parameter: --srcdir",
                "example": 'bbdev dc --srcdir arch/ReluBall_1 --top ReluBall',
            },
        )

    data = {
        "srcdir": srcdir,
        "top": body.get("top"),
        "keep_hierarchy": bool(body.get("keep_hierarchy", False)),
        "balltype": body.get("balltype"),
        "config": body.get("config", "sims.verilator.BuckyballToyVerilatorConfig"),
        "output_dir": body.get("output_dir"),
        "report_dir": f"{bbdir}/bb-tests/output/dc/reports",
    }
    await ctx.enqueue({"topic": "dc.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
