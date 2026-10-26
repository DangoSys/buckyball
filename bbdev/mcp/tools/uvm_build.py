"""MCP tool: bbdev_uvm_build."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_uvm_build(
        design: Optional[str] = None,
        ball: Optional[str] = None,
        ip: Optional[str] = None,
        device: bool = False,
    ) -> str:
        """Build a Ball or IP UVM simulation. POST /uvm/build."""
        if design is None and not ip:
            return err("Standalone UVM requires ip")
        params = {} if design is None else {"design": design}
        if ball and ip:
            return err("Parameters --ball and --ip are mutually exclusive")
        if ball:
            params["ball"] = ball
        if ip:
            params["ip"] = ip
        if device:
            params["device"] = True
        return fmt(submit("/uvm/build", params))
