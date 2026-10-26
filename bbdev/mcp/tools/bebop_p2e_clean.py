"""MCP tool: bbdev_bebop_p2e_clean."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_clean(
        design: str, device: bool = False, core: str | None = None
    ) -> str:
        """Clean bebop-p2e build. POST /bebop/p2e/clean."""
        if e := need("design", design):
            return err(e)
        request = {"design": design}
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(submit("/bebop/p2e/clean", request))
