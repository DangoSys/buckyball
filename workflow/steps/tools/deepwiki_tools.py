"""Deepwiki tool wrapper - via MCP Streamable HTTP"""

import httpx
import json
import uuid
from httpx_sse import connect_sse
from typing import Dict, Any
from .base import Tool


class DeepwikiAskTool(Tool):
    """Deepwiki Q&A tool - via MCP"""

    def get_name(self) -> str:
        return "deepwiki_ask"

    def get_description(self) -> str:
        return """Ask questions about a GitHub repository using Deepwiki.
    Use this to understand code, architecture, and implementation details."""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": (
                        "GitHub repository in format 'owner/repo' "
                        "(e.g., 'DangoSys/buckyball', 'ucb-bar/gemmini')"
                    ),
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask about the repository",
                },
            },
            "required": ["repo", "question"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        repo = arguments.get("repo")
        question = arguments.get("question")

        try:
            context.log_info(
                f"Asking Deepwiki via MCP HTTP: {question[:100]}... (repo: {repo})"
            )

            # MCP tool call request (JSON-RPC 2.0)
            request_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "ask_question",
                    "arguments": {"repoName": repo, "question": question},
                },
            }

            # Use Streamable HTTP endpoint
            mcp_url = "https://mcp.deepwiki.com/mcp"

            context.log_info(f"MCP URL: {mcp_url}")

            with httpx.Client(timeout=120.0) as client:
                # Step 1: Initialize session (without sessionId)
                init_payload = {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "buckyball-workflow",
                            "version": "1.0.0",
                        },
                    },
                }

                context.log_info("Initializing MCP session (without sessionId)...")

                session_id = None
                # Use POST request directly, get sessionId from response headers
                init_resp = client.post(
                    mcp_url,
                    json=init_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )

                # Get sessionId from response headers
                session_id = init_resp.headers.get("mcp-session-id")
                if not session_id:
                    context.log_error(f"Init response status: {init_resp.status_code}")
                    context.log_error(
                        f"Init response headers: {dict(init_resp.headers)}"
                    )
                    return "Error: No session ID in init response headers"

                context.log_info(f"Session initialized, ID: {session_id}")

                # Step 2: Call tool (with sessionId) - use SSE streaming
                context.log_info(
                    f"Calling tool: {json.dumps(request_payload, ensure_ascii=False)[:300]}"
                )

                # Use stream instead of connect_sse
                with client.stream(
                    "POST",
                    mcp_url,
                    json=request_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "Mcp-Session-Id": session_id,
                    },
                ) as tool_resp:
                    context.log_info(f"Tool response status: {tool_resp.status_code}")

                    # Manually parse SSE stream
                    for line in tool_resp.iter_lines():
                        if line.startswith("data: "):
                            # Remove "data: " prefix
                            data_str = line[6:]

                            # Skip heartbeat
                            if data_str.strip() == "ping":
                                continue

                            try:
                                result = json.loads(data_str)
                                context.log_info(
                                    f"Tool response: {json.dumps(result, ensure_ascii=False)[:500]}"
                                )

                                # Handle JSON-RPC 2.0 response
                                if "error" in result:
                                    error = result["error"]
                                    error_msg = f"MCP Error: {error.get('message', 'Unknown error')}"
                                    context.log_error(error_msg)
                                    return error_msg

                                if "result" in result:
                                    content = result["result"].get("content", [])
                                    if content and len(content) > 0:
                                        answer = content[0].get("text", "")
                                        context.log_info(
                                            f"Deepwiki answer length: {len(answer)} chars"
                                        )
                                        context.log_info(
                                            f"Deepwiki answer preview: {answer[:200]}..."
                                        )

                                        # Limit return length
                                        if len(answer) > 3000:
                                            answer = answer[:3000] + "\n... (truncated)"
                                        return answer
                            except json.JSONDecodeError as e:
                                context.log_error(
                                    f"Failed to parse SSE data: {e}, line: {line[:100]}"
                                )
                                continue

                    return "No valid response from Deepwiki"

        except Exception as e:
            error_msg = f"Error calling Deepwiki MCP: {str(e)}"
            context.log_error(error_msg)
            return error_msg


class DeepwikiReadWikiTool(Tool):
    """Read Deepwiki wiki content - via MCP"""

    def get_name(self) -> str:
        return "deepwiki_read_wiki"

    def get_description(self) -> str:
        return """Read wiki documentation for a GitHub repository from Deepwiki.
    Use this to get structured documentation about the repository."""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub repository in format 'owner/repo'",
                }
            },
            "required": ["repo"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        repo = arguments.get("repo")

        try:
            context.log_info(f"Reading Deepwiki wiki via MCP HTTP for: {repo}")

            # MCP tool call request
            request_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "read_wiki_contents",
                    "arguments": {"repoName": repo},
                },
            }

            # Use Streamable HTTP endpoint
            mcp_url = "https://mcp.deepwiki.com/mcp"

            context.log_info(f"MCP URL: {mcp_url}")

            with httpx.Client(timeout=120.0) as client:
                # Step 1: Initialize session (without sessionId)
                init_payload = {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "buckyball-workflow",
                            "version": "1.0.0",
                        },
                    },
                }

                context.log_info("Initializing MCP session (without sessionId)...")

                session_id = None
                # Use POST request directly, get sessionId from response headers
                init_resp = client.post(
                    mcp_url,
                    json=init_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )

                # Get sessionId from response headers
                session_id = init_resp.headers.get("mcp-session-id")
                if not session_id:
                    context.log_error(f"Init response status: {init_resp.status_code}")
                    context.log_error(
                        f"Init response headers: {dict(init_resp.headers)}"
                    )
                    return "Error: No session ID in init response headers"

                context.log_info(f"Session initialized, ID: {session_id}")

                # Step 2: Call tool (with sessionId) - use SSE streaming
                context.log_info(
                    f"Calling tool: {json.dumps(request_payload, ensure_ascii=False)}"
                )

                # Use stream instead of connect_sse
                with client.stream(
                    "POST",
                    mcp_url,
                    json=request_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "Mcp-Session-Id": session_id,
                    },
                ) as tool_resp:
                    context.log_info(f"Tool response status: {tool_resp.status_code}")

                    # Manually parse SSE stream
                    for line in tool_resp.iter_lines():
                        if line.startswith("data: "):
                            # Remove "data: " prefix
                            data_str = line[6:]

                            # Skip heartbeat
                            if data_str.strip() == "ping":
                                continue

                            try:
                                result = json.loads(data_str)
                                context.log_info(
                                    f"Tool response: {json.dumps(result, ensure_ascii=False)[:500]}"
                                )

                                # Handle JSON-RPC 2.0 response
                                if "error" in result:
                                    error = result["error"]
                                    error_msg = f"MCP Error: {error.get('message', 'Unknown error')}"
                                    context.log_error(error_msg)
                                    return error_msg

                                if "result" in result:
                                    content = result["result"].get("content", [])
                                    if content and len(content) > 0:
                                        wiki_text = content[0].get("text", "")
                                        context.log_info(
                                            f"Deepwiki wiki length: {len(wiki_text)} chars"
                                        )
                                        context.log_info(
                                            f"Deepwiki wiki preview: {wiki_text[:200]}..."
                                        )

                                        if len(wiki_text) > 5000:
                                            wiki_text = (
                                                wiki_text[:5000] + "\n... (truncated)"
                                            )
                                        return wiki_text
                            except json.JSONDecodeError as e:
                                context.log_error(
                                    f"Failed to parse SSE data: {e}, line: {line[:100]}"
                                )
                                continue

                    return "No valid wiki content from Deepwiki"

        except Exception as e:
            error_msg = f"Error reading Deepwiki wiki via MCP: {str(e)}"
            context.log_error(error_msg)
            return error_msg
