"""MCP tool: bbdev_verilator_run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import call, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_run(
        binary: str,
        config: str,
        batch: bool = True,
        coverage: bool = False,
        jobs: Optional[int] = None,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
    ) -> str:
        """Full non-bebop Verilator flow. POST /verilator/run."""
        if e := need("binary", binary):
            return err(e)
        if e := need("config", config):
            return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "config": config,
            "batch": batch,
            "coverage": coverage,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        if jobs is not None:
            params["jobs"] = jobs
        return fmt(call("/verilator/run", params, timeout=14400))
