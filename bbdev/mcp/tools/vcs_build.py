"""MCP tool: bbdev_vcs_build."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_vcs_build(
        jobs: int = 16,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Build the entry VCS simv. POST /vcs/build."""
        params: Dict[str, Any] = {}
        params["jobs"] = jobs
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/vcs/build", params))
