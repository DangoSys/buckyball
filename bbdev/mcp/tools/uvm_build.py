"""MCP tool: bbdev_uvm_build."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_uvm_build(
        config: str, ball: Optional[str] = None, filelist: Optional[str] = None
    ) -> str:
        """Build a Ball UVM simulation. POST /uvm/build."""
        if e := need("config", config):
            return err(e)
        params: Dict[str, Any] = {"config": config}
        if ball:
            params["ball"] = ball
        if filelist:
            params["filelist"] = filelist
        return fmt(call("/uvm/build", params, timeout=3600))

