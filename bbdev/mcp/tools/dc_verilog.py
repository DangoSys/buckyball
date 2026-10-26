"""MCP tool: bbdev_dc_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_dc_verilog(
        design: Optional[str] = None,
        top: Optional[str] = None,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Generate top-scoped DC RTL collateral. Defaults to DigitalTop."""
        if not design:
            return err("design is required")
        request = opt({}, design=design, top=top)
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/dc/verilog",
                request,
            )
        )
