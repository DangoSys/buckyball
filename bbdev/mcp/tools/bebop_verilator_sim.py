"""MCP tool: bbdev_bebop_verilator_sim."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_sim(
        binary: str,
        design: str,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        no_wave: bool = False,
        diff: bool = False,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Run one workload on bebop-verilator. POST /bebop/verilator/sim."""
        for n, v in (("binary", binary), ("design", design)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "design": design,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
            "no-wave": no_wave,
            "diff": diff,
        }
        if device is not None:
            params["device"] = device
        if core is not None:
            params["core"] = core
        return fmt(submit("/bebop/verilator/sim", params))
