"""MCP tool: bbdev_yosys_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_verilog(
        top: Optional[str] = None,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Generate Verilog for yosys flow. POST /yosys/verilog."""
        request = opt(
            {},
            top=top,
        )
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/yosys/verilog",
                request,
            )
        )
