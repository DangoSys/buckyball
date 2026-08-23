"""MCP tool: bbdev_yosys_synth."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_synth(
        chip: str,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        vcd: Optional[str] = None,
    ) -> str:
        """Yosys synthesis + OpenSTA. POST /yosys/synth."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/yosys/synth",
                opt(
                    {"chip": chip},
                    top=top,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    vcd=vcd,
                ),
            )
        )
