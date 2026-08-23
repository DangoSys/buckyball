"""MCP tool: bbdev_bebop_p2e_buildbitstream."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_buildbitstream(
        chip: str,
        vsrc_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Build bebop-p2e bitstream. POST /bebop/p2e/buildbitstream."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/bebop/p2e/buildbitstream",
                opt({"chip": chip}, vsrc_dir=vsrc_dir, output_dir=output_dir),
            )
        )
