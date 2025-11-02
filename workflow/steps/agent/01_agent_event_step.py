import httpx
import json
import os
from dotenv import load_dotenv
import sys

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.stream_run import stream_run_logger
from utils.event_common import check_result

load_dotenv()

config = {
    "type": "event",
    "name": "agent",
    "description": "Handle agent streaming response",
    "subscribes": ["agent.prompt"],
    "emits": ["agent.response"],
    "input": {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "model": {"type": "string"},
            "traceId": {"type": "string"},
            "apiKey": {"type": "string"},
            "baseUrl": {"type": "string"},
        },
    },
    "flows": ["agent"],
}


async def handler(input_data, context):
    context.logger.info("agent - Starting processing", {"input": input_data})

    message = input_data.get("message")
    model = input_data.get("model", "deepseek-chat")
    trace_id = input_data.get("traceId")

    # API configuration: prefer parameters passed in, otherwise use environment variables
    api_key = input_data.get("apiKey") or os.getenv("API_KEY")
    base_url = input_data.get("baseUrl") or os.getenv(
        "BASE_URL", "https://api.deepseek.com/v1"
    )

    if not api_key:
        error_msg = "API Key not provided"
        context.logger.error(error_msg)
        await context.emit(
            {
                "topic": "agent.error",
                "data": {
                    "error": error_msg,
                    "original_message": message,
                    "traceId": trace_id,
                },
            }
        )
        await check_result(context, 1, continue_run=False)
        return

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:

                if response.status_code != 200:
                    context.logger.error(f"agent API error: {response.status_code}")
                    return

                full_response = ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    full_response += content
                                    context.logger.info(f"{content}")

                        except json.JSONDecodeError:
                            continue

                # Send complete response
                await context.emit(
                    {
                        "topic": "agent.response",
                        "data": {
                            "response": full_response,
                            "original_message": message,
                            "traceId": trace_id,
                        },
                    }
                )

                context.logger.info(
                    "agent processing completed",
                    {"response_length": len(full_response), "traceId": trace_id},
                )

        # Pass response content back to API via extra_fields
        success_result, failure_result = await check_result(
            context, 0, continue_run=False, extra_fields={"response": full_response}
        )

    except Exception as e:
        context.logger.error(f"agent API call failed: {str(e)}")
        await context.emit(
            {
                "topic": "agent.error",
                "data": {
                    "error": str(e),
                    "original_message": message,
                    "traceId": trace_id,
                },
            }
        )

        # Pass error information back to API via extra_fields
        success_result, failure_result = await check_result(
            context, 1, continue_run=False, extra_fields={"error": str(e)}
        )

    # ==================================================================================
    #  finish workflow
    # ==================================================================================
    return
