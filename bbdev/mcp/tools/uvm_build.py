"""MCP tool: bbdev_uvm_build."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_uvm_build(
        ball: Optional[str] = None,
        ip: Optional[str] = None,
        config: str | None = None,
        device: str | None = None,
    ) -> str:
        """Build the UVM test of a Ball alone (ball: a prefab Ball or <design>/balls/<ball>), or of an IP
        (ip, with config and device). POST /uvm/build.
        """
        params = {}
        if ball and ip:
            return err("Parameters --ball and --ip are mutually exclusive")
        if ball:
            params["ball"] = ball
        if ip:
            params["ip"] = ip
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        return fmt(submit("/uvm/build", params))
