"""MCP tool: bbdev_yosys_run."""

from __future__ import annotations

from typing import Optional

from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_run(
        top: Optional[str] = None,
        config: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        vcd: Optional[str] = None,
    ) -> str:
        """Full yosys flow. POST /yosys/run."""
        return fmt(
            submit(
                "/yosys/run",
                opt(
                    {},
                    top=top,
                    config=config,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    vcd=vcd,
                ),
            )
        )
