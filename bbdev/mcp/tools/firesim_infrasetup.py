"""MCP tool: bbdev_firesim_infrasetup."""

from __future__ import annotations


from common import call, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_infrasetup(jobs: int = 16) -> str:
        """FireSim infrasetup. POST /firesim/infrasetup."""
        return fmt(call("/firesim/infrasetup", {"jobs": jobs}, timeout=7200))

