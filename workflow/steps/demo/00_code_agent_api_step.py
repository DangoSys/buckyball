import asyncio
from utils.event_common import wait_for_result
import sys
import os

config = {
    "type": "api",
    "name": "agent",
    "description": "通用 agent API，支持简化版 gemmini agent 和旧的多 agent 模式",
    "path": "/agent",
    "method": "POST",
    "emits": ["agent.prompt"],
    "bodySchema": {
        "type": "object",
        "properties": {
            "agentRole": {
                "type": "string",
                "description": "Agent 角色类型。使用 'gemmini' 启动简化版单一 Agent，其他值使用旧的多 Agent 模式",
                "default": "gemmini",
            },
            "promptPath": {
                "type": "string",
                "description": "用户需求 Markdown 文件路径。对于 gemmini agent，使用 prompt/gemmini_task.md",
            },
            "systemPromptPath": {
                "type": "string",
                "description": "系统角色 prompt 文件路径。对于 gemmini agent，使用 prompt/gemmini_ball_generator.md",
            },
            "workDir": {
                "type": "string",
                "description": "工作目录路径，默认为当前目录",
            },
            "model": {"type": "string", "default": "qwen3-235b-a22b-instruct-2507"},
            "apiKey": {"type": "string"},
            "baseUrl": {"type": "string"},
            "sessionId": {"type": "string", "description": "会话ID，用于多轮对话"},
        },
        "required": ["promptPath"],
    },
    "responseSchema": {
        "200": {
            "type": "object",
            "properties": {
                "traceId": {"type": "string"},
                "status": {"type": "string"},
                "agentRole": {"type": "string"},
                "files": {"type": "array", "description": "生成的文件列表"},
            },
        }
    },
    "flows": ["agent"],
}


async def handler(req, context):
    context.logger.info("agent API - 接收到请求", {"body": req.get("body")})
    body = req.get("body")

    agent_role = body.get("agentRole", "code")
    prompt_path = body.get("promptPath")
    system_prompt_path = body.get("systemPromptPath")
    work_dir = body.get("workDir", os.getcwd())
    model = body.get("model", "deepseek-chat")
    api_key = body.get("apiKey")
    base_url = body.get("baseUrl")
    session_id = body.get("sessionId")

    # 发送事件到处理步骤
    await context.emit(
        {
            "topic": "agent.prompt",
            "data": {
                "agentRole": agent_role,
                "promptPath": prompt_path,
                "systemPromptPath": system_prompt_path,
                "workDir": work_dir,
                "model": model,
                "traceId": context.trace_id,
                "apiKey": api_key,
                "baseUrl": base_url,
                "sessionId": session_id,
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
