from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "yosys-run",
    "description": "run yosys flow",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/run")],
    "enqueues": ["yosys.run"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}

    data = {
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
        "top": body.get("top"),
        "config": body.get("config"),
        "from_run_workflow": True,
    }
    await ctx.enqueue({"topic": "yosys.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
