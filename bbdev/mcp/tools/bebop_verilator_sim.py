"""MCP tool: bbdev_bebop_verilator_sim."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_sim(
        binary: str,
        config: str,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        no_wave: bool = False,
        log_dir: Optional[str] = None,
        fst_dir: Optional[str] = None,
    ) -> str:
        """Run one workload on bebop-verilator. POST /bebop/verilator/sim."""
        for n, v in (("binary", binary), ("config", config)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "binary": binary,
            "config": config,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
            "no-wave": no_wave,
        }
        if log_dir:
            params["log_dir"] = log_dir
        if fst_dir:
            params["fst_dir"] = fst_dir
        return fmt(call("/bebop/verilator/sim", params, timeout=7200))

