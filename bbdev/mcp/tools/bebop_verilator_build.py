"""MCP tool: bbdev_bebop_verilator_build."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_build(
        design: str,
        jobs: int = 16,
        diff: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Build bebop verilator binary. POST /bebop/verilator/build."""
        if e := need("design", design):
            return err(e)
        request = {"design": design, "jobs": jobs, "diff": diff}
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/bebop/verilator/build",
                request,
            )
        )
