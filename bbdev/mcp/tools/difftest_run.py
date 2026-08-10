"""MCP tool: bbdev_difftest_run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_difftest_run(
        backend: str,
        chip: Optional[str] = None,
        config: Optional[str] = None,
        binary: Optional[str] = None,
        jobs: int = 16,
        no_wave: bool = True,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        log_dir: Optional[str] = None,
    ) -> str:
        """Run Bank DiffTest on a selected backend using BEMU as reference. POST /difftest/run."""
        if error := need("backend", backend):
            return err(error)
        known_backends = ("verilator", "p2e")
        if backend not in known_backends:
            return err(
                f"Unsupported DiffTest backend: {backend}; "
                f"available backends: {', '.join(known_backends)}"
            )
        if backend == "verilator":
            for name, value in (("chip", chip), ("config", config), ("binary", binary)):
                if error := need(name, value):
                    return err(error)
        params: Dict[str, Any] = {
            "backend": backend,
            "jobs": jobs,
            "no-wave": no_wave,
            "itrace": itrace,
            "mtrace": mtrace,
            "pmctrace": pmctrace,
            "ctrace": ctrace,
            "banktrace": banktrace,
        }
        return fmt(
            submit(
                "/difftest/run",
                opt(
                    params,
                    chip=chip,
                    config=config,
                    binary=binary,
                    **{"log-dir": log_dir},
                ),
            )
        )
