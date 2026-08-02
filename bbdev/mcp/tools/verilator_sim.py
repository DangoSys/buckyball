"""MCP tool: bbdev_verilator_sim."""

from __future__ import annotations

from typing import Any, Dict

from common import call, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_sim(
        binary: str,
        config: str,
        batch: bool = True,
        coverage: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
    ) -> str:
        """Run one workload on non-bebop Verilator. POST /verilator/sim."""
        for n, v in (("binary", binary), ("config", config)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "config": config,
            "batch": batch,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        if coverage:
            params["coverage"] = True
        return fmt(call("/verilator/sim", params, timeout=7200))
