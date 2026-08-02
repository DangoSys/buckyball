"""MCP tool: bbdev_workload_tohex."""

from __future__ import annotations


from common import call, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_tohex() -> str:
        """Convert workload ELF to hex. POST /workload/tohex."""
        return fmt(call("/workload/tohex", {}, timeout=1800))

