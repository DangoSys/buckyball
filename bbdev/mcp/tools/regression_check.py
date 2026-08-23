"""MCP tool: bbdev_regression_check."""

from __future__ import annotations


from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_regression_check(
        chip: str,
        bitstream: str,
    ) -> str:
        """P2E pk-tests. POST /regression/check."""
        for n, v in (("chip", chip), ("bitstream", bitstream)):
            if e := need(n, v):
                return err(e)
        return fmt(
            submit(
                "/regression/check",
                {"chip": chip, "bitstream": bitstream},
            )
        )
