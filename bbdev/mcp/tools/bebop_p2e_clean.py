"""MCP tool: bbdev_bebop_p2e_clean."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_clean(config: str) -> str:
        """Clean bebop-p2e build. POST /bebop/p2e/clean."""
        if e := need("config", config):
            return err(e)
        return fmt(submit("/bebop/p2e/clean", {"config": config}))

