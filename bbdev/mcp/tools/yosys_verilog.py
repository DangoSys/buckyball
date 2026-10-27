"""MCP tool: bbdev_yosys_verilog."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_yosys_verilog(
        design: str,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Generate Verilog for yosys flow. POST /yosys/verilog."""
        if e := need("design", design):
            return err(e)
        request = opt(
            {"design": design},
            top=top,
            output_dir=output_dir,
        )
        if device is not None:
            request["device"] = device
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/yosys/verilog",
                request,
            )
        )
