"""MCP tool: bbdev_verilator_sim."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_sim(
        binary: str,
        chip: str,
        batch: bool = True,
        coverage: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
    ) -> str:
        """Run one workload on non-bebop Verilator. POST /verilator/sim."""
        for n, v in (("binary", binary), ("chip", chip)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "chip": chip,
            "batch": batch,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        if coverage:
            params["coverage"] = True
        return fmt(submit("/verilator/sim", params))
