"""MCP tool: bbdev_verilator_clean."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_clean(
        design: str, device: bool = False, core: str | None = None
    ) -> str:
        """Clean non-bebop Verilator build. POST /verilator/clean."""
        if e := need("design", design):
            return err(e)
        request = {"design": design}
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(submit("/verilator/clean", request))
