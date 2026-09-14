"""MCP tool: bbdev_firesim_buildbitstream."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_buildbitstream(chip: str) -> str:
        """Build FireSim bitstream. POST /firesim/buildbitstream."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/firesim/buildbitstream", {"chip": chip}))
