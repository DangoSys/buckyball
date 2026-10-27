"""FastMCP entry for buckyball-dev."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import register_all

mcp = FastMCP(
    "buckyball-dev",
    instructions=(
        "`device` names a device config registered in the design's"
        " configs/sims/default.toml [targets.device.configs], e.g."
        " PolyVerilatorConfig; `core` names a core simulation entry."
    ),
)
register_all(mcp)


def main() -> None:
    mcp.run(transport="stdio")
