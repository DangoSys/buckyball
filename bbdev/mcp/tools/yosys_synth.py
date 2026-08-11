"""MCP tool: bbdev_yosys_synth."""

from __future__ import annotations

from typing import Optional

from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_synth(
        top: Optional[str] = None,
        config: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        vcd: Optional[str] = None,
    ) -> str:
        """Yosys synthesis + OpenSTA. POST /yosys/synth."""
        return fmt(
            submit(
                "/yosys/synth",
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
