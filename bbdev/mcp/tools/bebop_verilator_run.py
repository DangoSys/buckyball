"""MCP tool: bbdev_bebop_verilator_run."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_run(
        binary: str,
        config: str,
        jobs: int = 16,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        no_wave: bool = False,
        rushB: bool = False,
        diff: bool = False,
        batch: bool = False,
    ) -> str:
        """Full bebop-verilator flow. POST /bebop/verilator/run."""
        for n, v in (("binary", binary), ("config", config)):
            if e := need(n, v):
                return err(e)
        return fmt(
            submit(
                "/bebop/verilator/run",
                {
                    "binary": binary,
                    "config": config,
                    "jobs": jobs,
                    "itrace": itrace,
                    "mtrace": mtrace,
                    "pmctrace": pmctrace,
                    "ctrace": ctrace,
                    "banktrace": banktrace,
                    "no-wave": no_wave,
                    "rushB": rushB,
                    "diff": diff,
                    "batch": batch,
                },
            )
        )
