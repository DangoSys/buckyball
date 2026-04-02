from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path
from utils.event_common import wait_for_result

config = {
    "name": "Yosys Synth",
    "description": "run yosys synthesis for area estimation",
    "flows": ["yosys"],
    "triggers": [api("POST", "/yosys/synth")],
    "enqueues": ["yosys.synth"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}

    data = {
        "output_dir": body.get("output_dir", f"{bbdir}/arch/build/"),
        "top": body.get("top"),
        "config": body.get("config"),
    }
    await ctx.enqueue({"topic": "yosys.synth", "data": {**data, "_trace_id": ctx.trace_id}})

    # ==================================================================================
    #  Wait for synthesis result
    # ==================================================================================
    result = await wait_for_result(ctx)
    return result
