"""MCP tool: bbdev_workload_clean."""

from __future__ import annotations


from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_clean() -> str:
        """Clean workload artifacts. POST /workload/clean."""
        return fmt(submit("/workload/clean", {}))

