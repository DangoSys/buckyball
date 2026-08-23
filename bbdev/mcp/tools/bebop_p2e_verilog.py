"""MCP tool: bbdev_bebop_p2e_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_verilog(
        chip: str, output_dir: Optional[str] = None
    ) -> str:
        """Generate Verilog for bebop-p2e. POST /bebop/p2e/verilog."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/bebop/p2e/verilog",
                opt({"chip": chip}, output_dir=output_dir),
            )
        )
