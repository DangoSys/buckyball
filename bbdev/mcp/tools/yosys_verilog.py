"""MCP tool: bbdev_yosys_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_verilog(
        top: Optional[str] = None,
        config: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ) -> str:
        """Generate Verilog for yosys flow. POST /yosys/verilog."""
        return fmt(
            submit(
                "/yosys/verilog",
                opt(
                    {},
                    top=top,
                    config=config,
                    output_dir=output_dir,
                    log_dir=log_dir,
                ),
            )
        )

