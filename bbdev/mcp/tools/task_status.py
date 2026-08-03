"""MCP tool: bbdev_task_status."""

from __future__ import annotations

from common import err, fmt, need, task_status


def register(mcp):
    @mcp.tool()
    def bbdev_task_status(trace_id: str) -> str:
        """Read a bbdev task result by trace_id without blocking."""
        if error := need("trace_id", trace_id):
            return err(error)
        try:
            return fmt(task_status(trace_id))
        except Exception as ex:
            return fmt({"success": False, "failure": True, "error": str(ex)})
