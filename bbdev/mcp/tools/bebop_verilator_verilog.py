"""MCP tool: bbdev_bebop_verilator_verilog."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_verilog(chip: str) -> str:
        """Generate Verilog for bebop-verilator. POST /bebop/verilator/verilog."""
        if e := need("chip", chip):
            return err(e)
        return fmt(submit("/bebop/verilator/verilog", {"chip": chip}))
