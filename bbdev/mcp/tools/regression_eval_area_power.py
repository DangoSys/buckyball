"""MCP tool: bbdev_regression_eval_area_power."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_regression_eval_area_power(
        chip: str,
        top: Optional[str] = None,
    ) -> str:
        """DC area+freq. POST /regression/eval-area-power."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/regression/eval-area-power",
                opt({"chip": chip}, top=top),
            )
        )
