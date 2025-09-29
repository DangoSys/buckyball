import asyncio
from utils.event_common import wait_for_result
import sys
import os


config = {
    "type": "api",
    "name": "doc_agent",
    "description": "生成代码目录文档",
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
    context.logger.info("doc-agent API - 接收到请求", {"body": req.get("body")})

    # 参数验证
    body = req.get("body", {})
    target_path = body.get("target_path")
    mode = body.get("mode")

    # 验证必需参数
    if not target_path:
        context.logger.error("doc-agent API - 缺少target_path参数")
        return {
            "status": 400,
            "body": {
                "error": "Missing required parameter",
                "details": "target_path is required",
            },
        }

    if not mode:
        context.logger.error("doc-agent API - 缺少mode参数")
        return {
            "status": 400,
            "body": {
                "error": "Missing required parameter",
                "details": "mode is required",
            },
        }

    # 验证mode参数值
    if mode not in ["create", "update"]:
        context.logger.error("doc-agent API - 无效的mode参数", {"mode": mode})
        return {
            "status": 400,
            "body": {
                "error": "Invalid parameter value",
                "details": "mode must be either 'create' or 'update'",
            },
        }

    # 验证target_path是否存在
    if not os.path.exists(target_path):
        context.logger.error(
            "doc-agent API - 目标路径不存在", {"target_path": target_path}
        )
        return {
            "status": 400,
            "body": {
                "error": "Invalid target path",
                "details": f"Path '{target_path}' does not exist",
            },
        }

    # 发送事件到处理步骤
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
    # 等待执行结果
    # ==================================================================================
    while True:
        result = await wait_for_result(context)
        if result is not None:
            return result
        await asyncio.sleep(1)
