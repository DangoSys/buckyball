"""MCP tool: bbdev_verilator_build."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_build(
        config: str,
        jobs: int = 16,
        coverage: bool = False,
    ) -> str:
        """Build non-bebop Verilator sim binary. POST /verilator/build."""
        if e := need("config", config):
            return err(e)
        params = {"config": config, "jobs": jobs}
        if coverage:
            params["coverage"] = True
        return fmt(submit("/verilator/build", params))
