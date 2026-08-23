"""MCP tool: bbdev_yosys_run."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_run(
        chip: str,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        vcd: Optional[str] = None,
    ) -> str:
        """Full yosys flow. POST /yosys/run."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/yosys/run",
                opt(
                    {"chip": chip},
                    top=top,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    vcd=vcd,
                ),
            )
        )
