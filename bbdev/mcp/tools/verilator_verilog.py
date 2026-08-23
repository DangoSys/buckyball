"""MCP tool: bbdev_verilator_verilog."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_verilog(chip: str) -> str:
        """Generate Verilog for non-bebop Verilator. POST /verilator/verilog."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/verilator/verilog", {"chip": chip}))
