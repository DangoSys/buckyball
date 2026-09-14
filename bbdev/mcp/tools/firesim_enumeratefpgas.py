"""MCP tool: bbdev_firesim_enumeratefpgas."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_enumeratefpgas(chip: str) -> str:
        """Enumerate FireSim FPGAs. POST /firesim/enumeratefpgas."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/firesim/enumeratefpgas", {"chip": chip}))
