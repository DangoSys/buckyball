"""MCP tool: bbdev_bemu_analysis."""

from __future__ import annotations

from pathlib import Path

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bemu_analysis(
        chip: str,
        log_dir: str,
        itrace: bool = False,
        mtrace: bool = False,
    ) -> str:
        """Analyze existing bebop-bemu traces. POST /bebop/bemu/analysis."""
        if e := need("chip", chip):
            return err(e)
        if e := need("log_dir", log_dir):
            return err(e)
        if not Path(log_dir).is_absolute():
            return err(f"log_dir must be an absolute path: {log_dir}")
        if not itrace and not mtrace:
            return err("need itrace and/or mtrace")
        return fmt(
            submit(
                "/bebop/bemu/analysis",
                {
                    "chip": chip,
                    "log-dir": log_dir,
                    "itrace": itrace,
                    "mtrace": mtrace,
                },
            )
        )
