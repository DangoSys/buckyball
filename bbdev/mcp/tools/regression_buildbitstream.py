"""MCP tool: bbdev_regression_buildbitstream."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_regression_buildbitstream(
        chip: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Build p2e bitstream. POST /regression/buildbitstream."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/regression/buildbitstream",
                opt({"chip": chip}, output_dir=output_dir),
            )
        )
