"""MCP tool: bbdev_workload_tohex."""

from __future__ import annotations

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_tohex(
        design: str, device: bool = False, core: str | None = None
    ) -> str:
        """Convert design workload ELF to hex. POST /workload/tohex."""
        request = {"design": design}
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(submit("/workload/tohex", request))
