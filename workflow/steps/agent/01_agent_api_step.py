import asyncio
from utils.event_common import wait_for_result
import sys
import os


config = {
    "type": "api",
    "name": "agent",
    "description": "Call agent API for streaming conversation",
    "path": "/agent/chat",
    "method": "POST",
    "emits": ["agent.prompt"],
    "bodySchema": {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "model": {"type": "string", "default": "deepseek-chat"},
            "apiKey": {"type": "string"},
            "baseUrl": {"type": "string"},
        },
        "required": ["message"],
    },
    "responseSchema": {
        "200": {
            "type": "object",
            "properties": {"traceId": {"type": "string"}, "status": {"type": "string"}},
        }
    },
    "flows": ["agent"],
}


async def handler(req, context):
    context.logger.info("agent API - Request received", {"body": req.get("body")})
    body = req.get("body")
    message = body.get("message")
    model = body.get("model", "deepseek-chat")
    api_key = body.get("apiKey")
    base_url = body.get("baseUrl")

    # Send event to processing step
    await context.emit(
        {
            "topic": "agent.prompt",
            "data": {
                "message": message,
                "model": model,
                "traceId": context.trace_id,
                "apiKey": api_key,
                "baseUrl": base_url,
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
