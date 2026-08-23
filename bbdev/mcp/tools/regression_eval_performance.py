"""MCP tool: bbdev_regression_eval_performance."""

from __future__ import annotations


from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_regression_eval_performance(
        chip: str,
        bitstream: str,
    ) -> str:
        """P2E model perfetto latency. POST /regression/eval-performance."""
        for n, v in (("chip", chip), ("bitstream", bitstream)):
            if e := need(n, v):
                return err(e)
        return fmt(
            submit(
                "/regression/eval-performance",
                {"chip": chip, "bitstream": bitstream},
            )
        )
