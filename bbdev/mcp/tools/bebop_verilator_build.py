"""MCP tool: bbdev_bebop_verilator_build."""

from __future__ import annotations


from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_build(config: str, jobs: int = 16) -> str:
        """Build bebop verilator binary. POST /bebop/verilator/build."""
        if e := need("config", config):
            return err(e)
        return fmt(
            call("/bebop/verilator/build", {"config": config, "jobs": jobs}, timeout=3600)
        )

