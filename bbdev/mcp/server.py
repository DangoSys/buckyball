"""FastMCP entry for buckyball-dev."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import register_all

mcp = FastMCP(
    "buckyball-dev",
    instructions=(
        "Exactly one of `device`, `hart`, `thread`, `ball` names the component to"
        " simulate, from its default config (a device: the one <Design>DefaultConfig.toml"
        " of its design). `config` picks another config: a config TOML Top relative to"
        " examples/designs, e.g. toy/devices/toy/configs/ToyGemvConfig.toml."
    ),
)
register_all(mcp)


def main() -> None:
    mcp.run(transport="stdio")
