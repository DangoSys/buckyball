"""MCP tool: bbdev_verilator_build."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_build(
        design: str,
        jobs: int = 16,
        coverage: bool = False,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Build non-bebop Verilator sim binary. POST /verilator/build."""
        if e := need("design", design):
            return err(e)
        params = {"design": design, "jobs": jobs}
        if coverage:
            params["coverage"] = True
        if device is not None:
            params["device"] = device
        if core is not None:
            params["core"] = core
        return fmt(submit("/verilator/build", params))
