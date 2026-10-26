"""MCP tool: bbdev_verilator_sim."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_sim(
        binary: str,
        design: str,
        batch: bool = False,
        coverage: bool = False,
        no_wave: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Run one workload on non-bebop Verilator. POST /verilator/sim."""
        for n, v in (("binary", binary), ("design", design)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "design": design,
            "batch": batch,
            "no-wave": no_wave,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        if coverage:
            params["coverage"] = True
        if device:
            params["device"] = True
        if core is not None:
            params["core"] = core
        return fmt(submit("/verilator/sim", params))
