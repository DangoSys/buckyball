"""MCP tool: bbdev_bebop_verilator_sim."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_sim(
        binary: str,
        chip: str,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        no_wave: bool = False,
        rushB: bool = False,
        batch: bool = False,
    ) -> str:
        """Run one workload on bebop-verilator. POST /bebop/verilator/sim."""
        for n, v in (("binary", binary), ("chip", chip)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "chip": chip,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
            "no-wave": no_wave,
            "rushB": rushB,
            "batch": batch,
        }
        return fmt(submit("/bebop/verilator/sim", params))
