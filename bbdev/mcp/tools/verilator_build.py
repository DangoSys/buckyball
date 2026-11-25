"""MCP tool: bbdev_verilator_build."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_build(
        jobs: int = 16,
        coverage: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Build non-bebop Verilator sim binary. POST /verilator/build."""
        params = {"jobs": jobs}
        if coverage:
            params["coverage"] = True
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/verilator/build", params))
