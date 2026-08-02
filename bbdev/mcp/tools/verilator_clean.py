"""MCP tool: bbdev_verilator_clean."""

from __future__ import annotations

from common import call, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_clean(config: str) -> str:
        """Clean non-bebop Verilator build. POST /verilator/clean."""
        if e := need("config", config):
            return err(e)
        return fmt(call("/verilator/clean", {"config": config}, timeout=300))
