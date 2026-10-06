"""MCP tool: bbdev_bebop_p2e_buildbitstream."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_buildbitstream(
        chip: str,
        diff: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        vsrc_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        stop_after: Optional[str] = None,
    ) -> str:
        """Build bebop-p2e bitstream. POST /bebop/p2e/buildbitstream."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/bebop/p2e/buildbitstream",
                opt(
                    {"chip": chip, "diff": diff, "itrace": itrace, "mtrace": mtrace},
                    vsrc_dir=vsrc_dir,
                    output_dir=output_dir,
                    stop_after=stop_after,
                ),
            )
        )
