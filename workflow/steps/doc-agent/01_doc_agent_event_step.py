import httpx
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 导入本地工具模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from doc_utils import detect_doc_type, load_prompt_template, prepare_update_mode_prompt

utils_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils"
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from utils.stream_run import stream_run_logger

load_dotenv()

config = {
    "type": "event",
    "name": "doc_agent",
    "description": "处理文档生成请求",
    "subscribes": ["doc.generate"],
    "emits": ["doc.response", "doc.integrate"],
    "input": {
        "type": "object",
        "properties": {
            "target_path": {"type": "string"},
            "mode": {"type": "string"},
            "traceId": {"type": "string"},
        },
    },
    "flows": ["doc_agent"],
}


async def handler(input_data, context):
    context.logger.info("doc-agent - 开始处理", {"input": input_data})

    target_path = input_data.get("target_path")
    mode = input_data.get("mode")
    trace_id = input_data.get("traceId")

    try:
        # 1. 检测文档类型并准备prompt
        doc_type = detect_doc_type(target_path)
        context.logger.info("doc-agent - 检测到文档类型", {"doc_type": doc_type})

        prompt_template = load_prompt_template(doc_type, target_path)
        prompt_template = prepare_update_mode_prompt(prompt_template, target_path, mode)

        # 2. 调用LLM API生成文档
        full_response = await generate_documentation(prompt_template, context)

        # 3. 保存文档
        output_path = os.path.join(target_path, "README.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_response)
        context.logger.info("doc-agent - 文档已保存", {"output_path": output_path})

        # 4. 发送集成事件
        await context.emit(
            {
                "topic": "doc.integrate",
                "data": {
                    "target_path": target_path,
                    "output_path": output_path,
                    "doc_type": doc_type,
                    "traceId": trace_id,
                },
            }
        )

        # 5. 发送完成响应
        await send_success_response(
            context, target_path, mode, doc_type, output_path, full_response, trace_id
        )

    except Exception as e:
        await send_error_response(context, str(e), target_path, mode, trace_id)


async def generate_documentation(prompt_template, context):
    """调用LLM API生成文档"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt_template}],
        "stream": True,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120.0,
        ) as response:

            if response.status_code != 200:
                error_text = await response.atext()
                raise Exception(f"API调用失败: {response.status_code}, {error_text}")

            full_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            full_response += content
                            context.logger.info(f"{content}")
                    except json.JSONDecodeError:
                        continue

    return full_response


async def send_success_response(
    context, target_path, mode, doc_type, output_path, full_response, trace_id
):
    """发送成功响应"""
    await context.emit(
        {
            "topic": "doc.response",
            "data": {
                "response": full_response,
                "target_path": target_path,
                "mode": mode,
                "doc_type": doc_type,
                "output_path": output_path,
                "traceId": trace_id,
            },
        }
    )

    success_result = {
        "status": 200,
        "body": {
            "success": True,
            "failure": False,
            "processing": False,
            "returncode": 0,
            "message": f"文档生成成功: {output_path}",
            "data": {
                "target_path": target_path,
                "mode": mode,
                "doc_type": doc_type,
                "output_path": output_path,
                "response_length": len(full_response),
            },
        },
    }
    await context.state.set(context.trace_id, "success", success_result)


async def send_error_response(context, error_msg, target_path, mode, trace_id):
    """发送错误响应"""
    full_error_msg = f"doc-agent处理失败: {error_msg}"
    context.logger.error(full_error_msg)

    await context.emit(
        {
            "topic": "doc.error",
            "data": {
                "error": full_error_msg,
                "target_path": target_path,
                "mode": mode,
                "traceId": trace_id,
            },
        }
    )

    failure_result = {
        "status": 500,
        "body": {
            "success": False,
            "failure": True,
            "processing": False,
            "returncode": 1,
            "error": full_error_msg,
        },
    }
    await context.state.set(context.trace_id, "failure", failure_result)
