"""MCP tool: bbdev_workload_clean."""

from __future__ import annotations

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_clean(
        design: str, device: str | None = None, core: str | None = None
    ) -> str:
        """Clean workload artifacts for one design. POST /workload/clean."""
        request = {"design": design}
        if device is not None:
            request["device"] = device
        if core is not None:
            request["core"] = core
        return fmt(submit("/workload/clean", request))
