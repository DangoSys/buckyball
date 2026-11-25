"""MCP tool: bbdev_bebop_verilator_clean."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_clean(
        config: str | None = None, device: str | None = None, hart: str | None = None
    ) -> str:
        """Clean bebop-verilator build. POST /bebop/verilator/clean."""
        request = {}
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(submit("/bebop/verilator/clean", request))
