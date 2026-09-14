"""MCP tool: bbdev_firesim_infrasetup."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_infrasetup(chip: str) -> str:
        """FireSim infrasetup. POST /firesim/infrasetup."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/firesim/infrasetup", {"chip": chip}))
