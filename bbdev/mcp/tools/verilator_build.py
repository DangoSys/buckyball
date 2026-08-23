"""MCP tool: bbdev_verilator_build."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_build(
        chip: str,
        jobs: int = 16,
        coverage: bool = False,
    ) -> str:
        """Build non-bebop Verilator sim binary. POST /verilator/build."""
        if e := need("chip", chip):
            return err(e)
        params = {"chip": chip, "jobs": jobs}
        if coverage:
            params["coverage"] = True
        return fmt(submit("/verilator/build", params))
