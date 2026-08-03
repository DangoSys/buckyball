"""MCP tool: bbdev_workload_tohex."""

from __future__ import annotations


from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_tohex() -> str:
        """Convert workload ELF to hex. POST /workload/tohex."""
        return fmt(submit("/workload/tohex", {}))

