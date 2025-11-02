import asyncio
from utils.event_common import wait_for_result
import sys
import os


config = {
    "type": "api",
    "name": "doc_agent",
    "description": "Generate code directory documentation",
    "path": "/doc/generate",
    "method": "POST",
    "emits": ["doc.generate"],
    "bodySchema": {
        "type": "object",
        "properties": {
            "target_path": {"type": "string"},
            "mode": {"type": "string", "enum": ["create", "update"]},
        },
        "required": ["target_path", "mode"],
    },
    "responseSchema": {
        "200": {
            "type": "object",
            "properties": {
                "traceId": {"type": "string"},
                "status": {"type": "string"},
                "message": {"type": "string"},
            },
        },
        "400": {
            "type": "object",
            "properties": {"error": {"type": "string"}, "details": {"type": "string"}},
        },
    },
    "flows": ["doc_agent"],
}


async def handler(req, context):
    context.logger.info("doc-agent API - Request received", {"body": req.get("body")})

    # Parameter validation
    body = req.get("body", {})
    target_path = body.get("target_path")
    mode = body.get("mode")

    # Validate required parameters
    if not target_path:
        context.logger.error("doc-agent API - Missing target_path parameter")
        return {
            "status": 400,
            "body": {
                "error": "Missing required parameter",
                "details": "target_path is required",
            },
        }

    if not mode:
        context.logger.error("doc-agent API - Missing mode parameter")
        return {
            "status": 400,
            "body": {
                "error": "Missing required parameter",
                "details": "mode is required",
            },
        }

    # Validate mode parameter value
    if mode not in ["create", "update"]:
        context.logger.error("doc-agent API - Invalid mode parameter", {"mode": mode})
        return {
            "status": 400,
            "body": {
                "error": "Invalid parameter value",
                "details": "mode must be either 'create' or 'update'",
            },
        }

    # Validate target_path exists
    if not os.path.exists(target_path):
        context.logger.error(
            "doc-agent API - Target path does not exist", {"target_path": target_path}
        )
        return {
            "status": 400,
            "body": {
                "error": "Invalid target path",
                "details": f"Path '{target_path}' does not exist",
            },
        }

    # Send event to processing step
    await context.emit(
        {
            "topic": "doc.generate",
            "data": {
                "target_path": target_path,
                "mode": mode,
                "traceId": context.trace_id,
            },
        }
    )

    # ==================================================================================
    # Wait for execution result
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
