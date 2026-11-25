"""MCP tool: bbdev_bebop_vcs_batch."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_vcs_batch(
        test: str,
        jobs: int = 16,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Run the design-owned VCS regression; test is "bare-tests" or "linux-tests". POST /bebop/vcs/batch."""
        params: Dict[str, Any] = {}
        params["test"] = test
        params["jobs"] = jobs
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/bebop/vcs/batch", params))
