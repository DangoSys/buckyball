"""MCP tool: bbdev_bebop_verilator_run."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_run(
        binary: str,
        design: str,
        jobs: int = 16,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        no_wave: bool = False,
        diff: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Full bebop-verilator flow. POST /bebop/verilator/run."""
        for n, v in (("binary", binary), ("design", design)):
            if e := need(n, v):
                return err(e)
        request = {
            "binary": binary,
            "design": design,
            "jobs": jobs,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
            "no-wave": no_wave,
            "diff": diff,
        }
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/bebop/verilator/run",
                request,
            )
        )
