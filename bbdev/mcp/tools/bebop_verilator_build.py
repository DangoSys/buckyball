"""MCP tool: bbdev_bebop_verilator_build."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_build(
        jobs: int = 16,
        diff: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Build bebop verilator binary. POST /bebop/verilator/build."""
        request = {"jobs": jobs, "diff": diff}
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/bebop/verilator/build",
                request,
            )
        )
