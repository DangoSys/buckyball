"""MCP tool: bbdev_bebop_verilator_build."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_build(
        chip: str, jobs: int = 16, diff: bool = False
    ) -> str:
        """Build bebop verilator binary. POST /bebop/verilator/build."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/bebop/verilator/build",
                {"chip": chip, "jobs": jobs, "diff": diff},
            )
        )
