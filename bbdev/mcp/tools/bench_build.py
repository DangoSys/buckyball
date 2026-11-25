"""MCP tool: bbdev_bench_build."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bench_build(
        bench: str,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Build the benchmarks of a simulation: bench "all" or one suite under
        bb-tests/workloads/src/benchmark (embench, coremark), each as one program
        <design>-<bench>-baremetal. POST /bench/build."""
        if e := need("bench", bench):
            return err(e)
        params: Dict[str, Any] = {"bench": bench}
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/bench/build", params))
