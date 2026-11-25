"""MCP tool: bbdev_bebop_verilator_run."""

from __future__ import annotations
from common import submit, err, fmt


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_run(
        model: str | None = None,
        bench: str | None = None,
        os: str | None = None,
        test: str | None = None,
        jobs: int = 16,
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
        """Full bebop-verilator flow. POST /bebop/verilator/run."""
        if sum(x is not None for x in (model, bench, test)) != 1:
            return err("give exactly one of model (with os), bench (with os) or test")
        request = {
            "jobs": jobs,
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
                request[key] = value
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/bebop/verilator/run",
                request,
            )
        )
