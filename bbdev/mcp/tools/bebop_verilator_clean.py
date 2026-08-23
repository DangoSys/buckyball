"""MCP tool: bbdev_bebop_verilator_clean."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_clean(chip: str) -> str:
        """Clean bebop-verilator build. POST /bebop/verilator/clean."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/bebop/verilator/clean", {"chip": chip}))
