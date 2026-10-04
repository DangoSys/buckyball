from pathlib import Path

from motia import ApiRequest, ApiResponse, FlowContext, api
from steps.bebop.p2e.scripts.runtime_case import validate_runtime_reuse

config = {
    "name": "bebop-p2e-runworkload-api",
    "description": "Run workload on FPGA via bebop p2e CLI",
    "flows": ["bebop"],
    "triggers": [api("POST", "/bebop/p2e/runworkload")],
    "enqueues": ["bebop.p2e.runworkload"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    if not isinstance(body.get("reuse-runtime", False), bool):
        return ApiResponse(status=400, body={"error": "reuse-runtime must be a boolean"})
    image = body.get("image", "")
    manifest = body.get("load-manifest", "")
    bitstream = body.get("bitstream", "")
    if not all(isinstance(value, str) for value in (image, manifest, bitstream)):
        return ApiResponse(status=400, body={"error": "image, load-manifest and bitstream must be strings"})
    if bool(image) == bool(manifest) or not bitstream:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "failure": True,
                "returncode": 400,
                "message": "exactly one of image and load-manifest, and bitstream, are required",
            },
        )
    if manifest and body.get("diff", False):
        return ApiResponse(status=400, body={"error": "load-manifest cannot use single-ELF diff"})
    if not Path(bitstream).is_file():
        return ApiResponse(status=400, body={"error": f"P2E bitstream is not a file: {bitstream}"})
    if body.get("reuse-runtime", False):
        try:
            validate_runtime_reuse(bitstream, bool(body.get("diff", False)))
        except (ValueError, OSError) as error:
            return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue({
        "topic": "bebop.p2e.runworkload",
        "data": {**body, "_trace_id": ctx.trace_id},
    })
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
