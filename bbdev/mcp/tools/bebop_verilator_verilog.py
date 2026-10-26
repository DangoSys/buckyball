"""MCP tool: bbdev_bebop_verilator_verilog."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_verilog(
        design: str, diff: bool = False, device: bool = False, core: str | None = None
    ) -> str:
        """Generate Verilog for bebop-verilator. POST /bebop/verilator/verilog."""
        if e := need("design", design):
            return err(e)
        request = {"design": design, "diff": diff}
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(submit("/bebop/verilator/verilog", request))
