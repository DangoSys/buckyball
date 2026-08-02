"""MCP tool: bbdev_dc_verilog."""

from __future__ import annotations

from typing import Optional

from common import call, err, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_dc_verilog(
        config: Optional[str] = None, rtl_dir: Optional[str] = None
    ) -> str:
        """Generate Verilog for DC flow. POST /dc/verilog (rtl_dir maps to dir)."""
        if not config and not rtl_dir:
            return err("config or rtl_dir is required")
        return fmt(
            call(
                "/dc/verilog",
                opt({}, config=config, dir=rtl_dir),
                timeout=1800,
            )
        )

