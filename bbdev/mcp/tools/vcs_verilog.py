"""MCP tool: bbdev_vcs_verilog."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_vcs_verilog(
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Generate the entry tapeout RTL for VCS. POST /vcs/verilog."""
        params: Dict[str, Any] = {}
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/vcs/verilog", params))
