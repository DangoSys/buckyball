"""MCP tool: bbdev_bebop_p2e_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_verilog(
        design: str,
        output_dir: Optional[str] = None,
        diff: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Generate Verilog for bebop-p2e. POST /bebop/p2e/verilog."""
        if e := need("design", design):
            return err(e)
        request = opt({"design": design, "diff": diff}, output_dir=output_dir)
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/bebop/p2e/verilog",
                request,
            )
        )
