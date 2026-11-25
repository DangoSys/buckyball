"""MCP tool: bbdev_yosys_synth."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_synth(
        top: Optional[str] = None,
        vcd: Optional[str] = None,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Yosys synthesis + OpenSTA. POST /yosys/synth."""
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
                "/yosys/synth",
                request,
            )
        )
