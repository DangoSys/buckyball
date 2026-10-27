"""MCP tool: bbdev_verilator_verilog."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_verilog(
        design: str, device: str | None = None, core: str | None = None
    ) -> str:
        """Generate Verilog for non-bebop Verilator. POST /verilator/verilog."""
        if e := need("design", design):
            return err(e)
        request = {"design": design}
        if device is not None:
            request["device"] = device
        if core is not None:
            request["core"] = core
        return fmt(submit("/verilator/verilog", request))
