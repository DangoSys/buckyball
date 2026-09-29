import os
import sys

from motia import ApiRequest, ApiResponse, FlowContext, api

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from steps.model.scripts.build import validate_model_build
from utils.path import get_buckyball_path

config = {
    "name": "model-build-api",
    "description": "Build a chip model recipe",
    "flows": ["model"],
    "triggers": [api("POST", "/model/build")],
    "enqueues": ["model.build"],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    try:
        validate_model_build(get_buckyball_path(), body)
    except ValueError as error:
        return ApiResponse(status=400, body={"error": str(error)})
    await ctx.enqueue({"topic": "model.build", "data": {**body, "_trace_id": ctx.trace_id}})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
