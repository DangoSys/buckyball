"""MCP tool: bbdev_firesim_runworkload."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_firesim_runworkload(
        design: str, device: bool = False, core: str | None = None
    ) -> str:
        """Run FireSim workload. POST /firesim/runworkload."""
        if e := need("design", design):
            return err(e)
        request = {"design": design}
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(submit("/firesim/runworkload", request))
