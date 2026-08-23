"""MCP tool: bbdev_yosys_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_verilog(
        chip: str,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ) -> str:
        """Generate Verilog for yosys flow. POST /yosys/verilog."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/yosys/verilog",
                opt(
                    {"chip": chip},
                    top=top,
                    output_dir=output_dir,
                    log_dir=log_dir,
                ),
            )
        )
