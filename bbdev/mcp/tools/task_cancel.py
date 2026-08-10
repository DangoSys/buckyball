"""MCP tool: bbdev_task_cancel."""

from __future__ import annotations

from common import err, fmt, need, submit


def register(mcp):
    @mcp.tool()
    def bbdev_task_cancel(trace_id: str) -> str:
        """Cancel a queued or running bbdev task and terminate its process group."""
        if error := need("trace_id", trace_id):
            return err(error)
        try:
            return fmt(submit(f"/task/{trace_id}/cancel", {}))
        except Exception as ex:
            return fmt({"success": False, "failure": True, "error": str(ex)})
