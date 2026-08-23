"""MCP tool: bbdev_workload_clean."""

from __future__ import annotations

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_clean(chip: str) -> str:
        """Clean workload artifacts for one chip. POST /workload/clean."""
        return fmt(submit("/workload/clean", {"chip": chip}))
