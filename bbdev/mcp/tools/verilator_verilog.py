"""MCP tool: bbdev_verilator_verilog."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_verilog(
        config: str | None = None, device: str | None = None, hart: str | None = None
    ) -> str:
        """Generate Verilog for non-bebop Verilator. POST /verilator/verilog."""
        request = {}
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(submit("/verilator/verilog", request))
