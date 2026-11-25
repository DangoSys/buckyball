"""MCP tool: bbdev_bebop_vcs_verilog."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_vcs_verilog(
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Generate Bebop VCS RTL once. POST /bebop/vcs/verilog."""
        params: Dict[str, Any] = {}
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/bebop/vcs/verilog", params))
