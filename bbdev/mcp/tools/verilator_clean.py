"""MCP tool: bbdev_verilator_clean."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_clean(chip: str) -> str:
        """Clean non-bebop Verilator build. POST /verilator/clean."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/verilator/clean", {"chip": chip}))
