import os

from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "regression-check-api",
    "description": "Run p2e pk-tests and record accuracy for regression",
    "flows": ["regression"],
    "triggers": [api("POST", "/regression/check")],
    "enqueues": ["regression.check"],
}


def _req_str(body: dict, key: str):
    value = body.get(key)
    if not isinstance(value, str) or not value or value == "None":
        return None
    return value


async def handler(req: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = req.body or {}
    chip = _req_str(body, "chip")
    if chip is None:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --chip must be specified"},
        )
    bitstream = _req_str(body, "bitstream")
    if bitstream is None:
        return ApiResponse(
            status=400,
            body={"error": "Missing required parameter: --bitstream must be specified"},
        )
    if not os.path.isfile(bitstream):
        return ApiResponse(
            status=400,
            body={"error": f"bitstream file not found: {bitstream}"},
        )

    data = {
        "chip": chip,
        "bitstream": bitstream,
        "_trace_id": ctx.trace_id,
    }

    await ctx.enqueue({"topic": "regression.check", "data": data})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
