"""MCP tool: bbdev_uvm_build."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_uvm_build(chip: str, ball: Optional[str] = None) -> str:
        """Build a Ball UVM simulation. POST /uvm/build."""
        if e := need("chip", chip):
            return err(e)
        params = {"chip": chip}
        if ball:
            params["ball"] = ball
        return fmt(submit("/uvm/build", params))
