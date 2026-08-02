"""MCP tool: bbdev_bebop_p2e_verilog."""

from __future__ import annotations

from typing import Optional

from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_verilog(
        config: str, output_dir: Optional[str] = None
    ) -> str:
        """Generate Verilog for bebop-p2e. POST /bebop/p2e/verilog."""
        if e := need("config", config):
            return err(e)
        return fmt(
            call(
                "/bebop/p2e/verilog",
                opt({"config": config}, output_dir=output_dir),
                timeout=1800,
            )
        )

