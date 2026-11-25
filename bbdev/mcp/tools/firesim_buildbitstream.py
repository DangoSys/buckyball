"""MCP tool: bbdev_firesim_buildbitstream."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_buildbitstream(
        config: str | None = None, device: str | None = None, hart: str | None = None
    ) -> str:
        """Build FireSim bitstream. POST /firesim/buildbitstream."""
        request = {}
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(submit("/firesim/buildbitstream", request))
