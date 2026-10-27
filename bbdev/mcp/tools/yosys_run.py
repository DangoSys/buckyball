"""MCP tool: bbdev_yosys_run."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_run(
        design: str,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        vcd: Optional[str] = None,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Full yosys flow. POST /yosys/run."""
        if e := need("design", design):
            return err(e)
        request = opt(
            {"design": design},
            top=top,
            output_dir=output_dir,
            vcd=vcd,
        )
        if device is not None:
            request["device"] = device
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/yosys/run",
                request,
            )
        )
