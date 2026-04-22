from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path

config = {
    "name": "dc-run",
    "description": "run Design Compiler synthesis",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/run")],
    "enqueues": ["dc.run"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}

    data = {
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
        "top": body.get("top"),
        "config": body.get("config"),
    }
    await ctx.enqueue({"topic": "dc.run", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
