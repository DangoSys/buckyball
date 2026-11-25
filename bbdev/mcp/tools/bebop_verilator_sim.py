"""MCP tool: bbdev_bebop_verilator_sim."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_sim(
        model: str | None = None,
        bench: str | None = None,
        os: str | None = None,
        test: str | None = None,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        no_wave: bool = False,
        diff: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Run one workload on bebop-verilator. POST /bebop/verilator/sim."""
        if sum(x is not None for x in (model, bench, test)) != 1:
            return err("give exactly one of model (with os), bench (with os) or test")
        params: Dict[str, Any] = {
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
            "no-wave": no_wave,
            "diff": diff,
        }
        for key, value in (
            ("model", model),
            ("bench", bench),
            ("os", os),
            ("test", test),
        ):
            if value is not None:
                params[key] = value
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/bebop/verilator/sim", params))
