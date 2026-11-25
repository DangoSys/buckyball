"""MCP tool: bbdev_yosys_run."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_run(
        top: Optional[str] = None,
        vcd: Optional[str] = None,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Full yosys flow. POST /yosys/run."""
        request = opt(
            {},
            top=top,
            vcd=vcd,
        )
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/yosys/run",
                request,
            )
        )
