"""MCP tool: bbdev_verilator_run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_run(
        binary: str,
        design: str,
        batch: bool = False,
        coverage: bool = False,
        no_wave: bool = False,
        jobs: Optional[int] = None,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Full non-bebop Verilator flow. POST /verilator/run."""
        if e := need("binary", binary):
            return err(e)
        if e := need("design", design):
            return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "design": design,
            "batch": batch,
            "coverage": coverage,
            "no-wave": no_wave,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        if jobs is not None:
            params["jobs"] = jobs
        if device is not None:
            params["device"] = device
        if core is not None:
            params["core"] = core
        return fmt(submit("/verilator/run", params))
