"""MCP tool: bbdev_bebop_p2e_buildbitstream."""

from __future__ import annotations

from typing import Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_buildbitstream(
        diff: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Build bebop-p2e bitstream. POST /bebop/p2e/buildbitstream."""
        request = opt(
            {"diff": diff, "itrace": itrace, "mtrace": mtrace},
        )
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/bebop/p2e/buildbitstream",
                request,
            )
        )
