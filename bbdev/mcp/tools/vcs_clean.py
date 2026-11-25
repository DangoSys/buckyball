"""MCP tool: bbdev_vcs_clean."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_vcs_clean(
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Clean the entry VCS build. POST /vcs/clean."""
        params: Dict[str, Any] = {}
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/vcs/clean", params))
