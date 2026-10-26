"""MCP tool: bbdev_bemu_analysis."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bemu_analysis(
        design: str,
        run: str,
        itrace: bool = False,
        mtrace: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Analyze a run directory under the repository log/ directory. POST /bebop/bemu/analysis."""
        if e := need("design", design):
            return err(e)
        if e := need("run", run):
            return err(e)
        if not itrace and not mtrace:
            return err("need itrace and/or mtrace")
        request = {
            "design": design,
            "run": run,
            "itrace": itrace,
            "mtrace": mtrace,
        }
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/bebop/bemu/analysis",
                request,
            )
        )
