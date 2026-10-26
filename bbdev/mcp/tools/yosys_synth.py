"""MCP tool: bbdev_yosys_synth."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_synth(
        design: str,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        vcd: Optional[str] = None,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Yosys synthesis + OpenSTA. POST /yosys/synth."""
        if e := need("design", design):
            return err(e)
        request = opt(
            {"design": design},
            top=top,
            output_dir=output_dir,
            vcd=vcd,
        )
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/yosys/synth",
                request,
            )
        )
