"""MCP tool: bbdev_uvm_build."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_uvm_build(
        chip: str, ball: Optional[str] = None, ip: Optional[str] = None
    ) -> str:
        """Build a Ball or IP UVM simulation. POST /uvm/build."""
        if e := need("chip", chip):
            return err(e)
        params = {"chip": chip}
        if ball and ip:
            return err("Parameters --ball and --ip are mutually exclusive")
        if ball:
            params["ball"] = ball
        if ip:
            params["ip"] = ip
        return fmt(submit("/uvm/build", params))
