"""FastMCP entry for buckyball-dev."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import register_all

mcp = FastMCP("buckyball-dev")
register_all(mcp)


def main() -> None:
    mcp.run(transport="stdio")
