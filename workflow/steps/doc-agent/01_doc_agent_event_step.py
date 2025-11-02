import httpx
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import local utility modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from doc_utils import detect_doc_type, load_prompt_template, prepare_update_mode_prompt

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result

load_dotenv()

config = {
    "type": "event",
    "name": "doc_agent",
    "description": "Handle documentation generation requests",
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
    context.logger.info("doc-agent - Start processing", {"input": input_data})

    target_path = input_data.get("target_path")
    mode = input_data.get("mode")
    trace_id = input_data.get("traceId")

    try:
        # 1. Detect document type and prepare prompt
        doc_type = detect_doc_type(target_path)
        context.logger.info("doc-agent - Document type detected", {"doc_type": doc_type})

        prompt_template = load_prompt_template(doc_type, target_path)
        prompt_template = prepare_update_mode_prompt(prompt_template, target_path, mode)

        # 2. Call LLM API to generate documentation
        full_response = await generate_documentation(prompt_template, context)

        # 3. Save documentation
        output_path = os.path.join(target_path, "README.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_response)
        context.logger.info("doc-agent - Documentation saved", {"output_path": output_path})

        # 4. Send integration event
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

        # 5. Send completion response
        await send_success_response(
            context, target_path, mode, doc_type, output_path, full_response, trace_id
        )

    except Exception as e:
        await send_error_response(context, str(e), target_path, mode, trace_id)


async def generate_documentation(prompt_template, context):
    """Call LLM API to generate documentation"""
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
                raise Exception(f"API call failed: {response.status_code}, {error_text}")

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
    """Send success response"""
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

    success_result, failure_result = await check_result(
        context,
        0,
        continue_run=False,
        extra_fields={
            "message": f"Documentation generation successful: {output_path}",
            "data": {
                "target_path": target_path,
                "mode": mode,
                "doc_type": doc_type,
                "output_path": output_path,
                "response_length": len(full_response),
            },
        },
    )


async def send_error_response(context, error_msg, target_path, mode, trace_id):
    """Send error response"""
    full_error_msg = f"doc-agent processing failed: {error_msg}"
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

    success_result, failure_result = await check_result(
        context,
        1,
        continue_run=False,
        extra_fields={
            "error": full_error_msg,
        },
    )
