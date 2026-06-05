from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import check_dc_rtl_args, get_buckyball_path, get_dc_rtl_dir

config = {
    "name": "dc-verilog-api",
    "description": "generate verilog for dc flow",
    "flows": ["dc"],
    "triggers": [api("POST", "/dc/verilog")],
    "enqueues": ["dc.verilog"],
}


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    bbdir = get_buckyball_path()
    body = req.body or {}
    try:
        check_dc_rtl_args(body)
        output_dir = get_dc_rtl_dir(bbdir, body.get("config"), body.get("dir"))
    except ValueError as e:
        return ApiResponse(status=400, body={"error": str(e)})

    data = {
        "output_dir": output_dir,
        "config": body.get("config"),
    }
    await ctx.enqueue({"topic": "dc.verilog", "data": {**data, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
