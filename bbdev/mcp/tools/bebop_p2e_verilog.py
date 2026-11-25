"""MCP tool: bbdev_bebop_p2e_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_verilog(
        diff: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Generate Verilog for bebop-p2e. POST /bebop/p2e/verilog."""
        request = opt({"diff": diff})
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/bebop/p2e/verilog",
                request,
            )
        )
