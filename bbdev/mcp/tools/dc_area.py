"""MCP tool: bbdev_dc_area."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_dc_area(
        top: Optional[str] = None,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Run DC synthesis and area reporting. Defaults to DigitalTop."""
        request = opt({}, top=top)
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(submit("/dc/area", request))
