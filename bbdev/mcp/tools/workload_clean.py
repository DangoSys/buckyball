"""MCP tool: bbdev_workload_clean."""

from __future__ import annotations

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_clean(
        config: str | None = None, device: str | None = None, hart: str | None = None
    ) -> str:
        """Clean workload artifacts for one design. POST /workload/clean."""
        request = {}
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(submit("/workload/clean", request))
