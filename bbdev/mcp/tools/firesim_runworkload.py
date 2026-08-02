"""MCP tool: bbdev_firesim_runworkload."""

from __future__ import annotations


from common import call, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_runworkload(jobs: int = 16) -> str:
        """Run FireSim workload. POST /firesim/runworkload."""
        return fmt(call("/firesim/runworkload", {"jobs": jobs}, timeout=14400))

