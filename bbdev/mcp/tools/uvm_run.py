"""MCP tool: bbdev_uvm_run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_uvm_run(
        ball: str,
        filelist: Optional[str] = None,
        test: Optional[str] = None,
    ) -> str:
        """Build and run a Ball UVM simulation. POST /uvm/run."""
        if e := need("ball", ball):
            return err(e)
        params: Dict[str, Any] = {"ball": ball}
        if filelist:
            params["filelist"] = filelist
        if test:
            params["test"] = test
        return fmt(submit("/uvm/run", params))

