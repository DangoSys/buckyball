"""MCP tool: bbdev_dc_area."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_dc_area(config: Optional[str] = None, top: Optional[str] = None) -> str:
        """Run DC synthesis and area reporting. Defaults to DigitalTop."""
        if not config:
            return err("config is required")
        return fmt(submit("/dc/area", opt({}, config=config, top=top)))
