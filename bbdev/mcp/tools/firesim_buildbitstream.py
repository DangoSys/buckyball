"""MCP tool: bbdev_firesim_buildbitstream."""

from __future__ import annotations


from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_buildbitstream() -> str:
        """Build FireSim bitstream. POST /firesim/buildbitstream."""
        return fmt(submit("/firesim/buildbitstream", {}))

