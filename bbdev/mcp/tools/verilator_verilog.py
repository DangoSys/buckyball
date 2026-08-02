"""MCP tool: bbdev_verilator_verilog."""

from __future__ import annotations

from common import call, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_verilog(config: str) -> str:
        """Generate Verilog for non-bebop Verilator. POST /verilator/verilog."""
        if e := need("config", config):
            return err(e)
        return fmt(call("/verilator/verilog", {"config": config}, timeout=1800))
