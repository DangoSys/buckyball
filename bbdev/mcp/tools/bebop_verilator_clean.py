"""MCP tool: bbdev_bebop_verilator_clean."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_clean(
        design: str, device: str | None = None, core: str | None = None
    ) -> str:
        """Clean bebop-verilator build. POST /bebop/verilator/clean."""
        if e := need("design", design):
            return err(e)
        request = {"design": design}
        if device is not None:
            request["device"] = device
        if core is not None:
            request["core"] = core
        return fmt(submit("/bebop/verilator/clean", request))
