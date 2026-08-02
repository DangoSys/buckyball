"""MCP tool: bbdev_firesim_enumeratefpgas."""

from __future__ import annotations


from common import call, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_enumeratefpgas() -> str:
        """Enumerate FireSim FPGAs. POST /firesim/enumeratefpgas."""
        return fmt(call("/firesim/enumeratefpgas", {}, timeout=1800))

