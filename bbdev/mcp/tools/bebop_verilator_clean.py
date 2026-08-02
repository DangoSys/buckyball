"""MCP tool: bbdev_bebop_verilator_clean."""

from __future__ import annotations


from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_clean(config: str) -> str:
        """Clean bebop-verilator build. POST /bebop/verilator/clean."""
        if e := need("config", config):
            return err(e)
        return fmt(call("/bebop/verilator/clean", {"config": config}, timeout=300))

