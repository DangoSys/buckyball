"""MCP tool: bbdev_dc_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_dc_verilog(
        config: Optional[str] = None,
        top: Optional[str] = None,
    ) -> str:
        """Generate top-scoped DC RTL collateral. Defaults to DigitalTop."""
        if not config:
            return err("config is required")
        return fmt(
            submit(
                "/dc/verilog",
                opt({}, config=config, top=top),
            )
        )
