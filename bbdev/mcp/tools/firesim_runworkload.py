"""MCP tool: bbdev_firesim_runworkload."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_runworkload(chip: str) -> str:
        """Run FireSim workload. POST /firesim/runworkload."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/firesim/runworkload", {"chip": chip}))
