"""MCP tool: bbdev_bebop_verilator_verilog."""

from __future__ import annotations


from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_verilog(config: str) -> str:
        """Generate Verilog for bebop-verilator. POST /bebop/verilator/verilog."""
        if e := need("config", config):
            return err(e)
        return fmt(call("/bebop/verilator/verilog", {"config": config}, timeout=1800))

