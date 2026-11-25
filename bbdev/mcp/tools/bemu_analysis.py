"""MCP tool: bbdev_bemu_analysis."""

from __future__ import annotations

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bemu_analysis(
        run: str,
        itrace: bool = False,
        mtrace: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Analyze a run directory under the repository log/ directory. POST /bebop/bemu/analysis."""
        if e := need("run", run):
            return err(e)
        if not itrace and not mtrace:
            return err("need itrace and/or mtrace")
        request = {
            "run": run,
            "itrace": itrace,
            "mtrace": mtrace,
        }
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/bebop/bemu/analysis",
                request,
            )
        )
