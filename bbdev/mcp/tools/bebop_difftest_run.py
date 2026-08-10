"""MCP tool: bbdev_bebop_difftest_run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_difftest_run(
        chip: str,
        config: str,
        binary: str,
        jobs: int = 16,
        no_wave: bool = True,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        log_dir: Optional[str] = None,
    ) -> str:
        """Build and run Verilator+BEMU Bank DiffTest. POST /bebop/difftest/run."""
        for name, value in (("chip", chip), ("config", config), ("binary", binary)):
            if error := need(name, value):
                return err(error)
        params: Dict[str, Any] = {
            "chip": chip,
            "config": config,
            "binary": binary,
            "jobs": jobs,
            "no-wave": no_wave,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        return fmt(submit("/bebop/difftest/run", opt(params, **{"log-dir": log_dir})))
