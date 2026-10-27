"""MCP tool: bbdev_bebop_p2e_buildbitstream."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_buildbitstream(
        design: str,
        diff: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        vsrc_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Build bebop-p2e bitstream. POST /bebop/p2e/buildbitstream."""
        if e := need("design", design):
            return err(e)
        request = opt(
            {"design": design, "diff": diff, "itrace": itrace, "mtrace": mtrace},
            vsrc_dir=vsrc_dir,
            output_dir=output_dir,
        )
        if device is not None:
            request["device"] = device
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/bebop/p2e/buildbitstream",
                request,
            )
        )
