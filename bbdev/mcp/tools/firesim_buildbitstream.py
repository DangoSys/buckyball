"""MCP tool: bbdev_firesim_buildbitstream."""

from __future__ import annotations


from common import call, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_buildbitstream() -> str:
        """Build FireSim bitstream. POST /firesim/buildbitstream."""
        return fmt(call("/firesim/buildbitstream", {}, timeout=14400))

