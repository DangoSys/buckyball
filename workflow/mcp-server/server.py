#!/usr/bin/env python3
"""BuckyBall Development Tools MCP Server"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
workflow_dir = Path(__file__).parent.parent
sys.path.insert(0, str(workflow_dir))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools import get_all_tools, handle_tool_call

# Create server
server = Server("bbdev-mcp")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools."""
    return get_all_tools()


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool calls."""
    try:
        result = await handle_tool_call(name, arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
