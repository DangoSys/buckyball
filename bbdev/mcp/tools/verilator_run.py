"""MCP tool: bbdev_verilator_run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, err, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_verilator_run(
        model: str | None = None,
        bench: str | None = None,
        os: str | None = None,
        test: str | None = None,
        batch: bool = False,
        coverage: bool = False,
        no_wave: bool = False,
        jobs: Optional[int] = None,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Full non-bebop Verilator flow. POST /verilator/run."""
        if sum(x is not None for x in (model, bench, test)) != 1:
            return err("give exactly one of model (with os), bench (with os) or test")
        params: Dict[str, Any] = {
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
        return fmt(submit("/verilator/run", params))
