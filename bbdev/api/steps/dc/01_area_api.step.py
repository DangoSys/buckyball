from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import check_dc_rtl_args, get_buckyball_path, get_dc_analysis_dir, get_dc_rtl_dir

config = {
    "name": "dc-area-api",
    "description": "run DC synthesis and area reports",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/area")],
    "enqueues": ["dc.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}
    try:
        check_dc_rtl_args(body)
        rtl_dir = get_dc_rtl_dir(bbdir, body.get("config"))
        analysis_dir = get_dc_analysis_dir(bbdir, body.get("config"), "area")
    except ValueError as exc:
        return ApiResponse(status=400, body={"error": str(exc)})
    await ctx.enqueue(
        {
            "topic": "dc.verilog",
            "data": {
                "output_dir": rtl_dir,
                "analysis_dir": analysis_dir,
                "config": body.get("config"),
                "top": body.get("top") or "DigitalTop",
                "from_area_workflow": True,
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
